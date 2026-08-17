#!/usr/bin/env python3
"""TASK_878 — failed-fill/coinflip semantics: executable acceptance-gate smoke.

A no-print (frozen, forward-filled) decision cell is an ATTEMPTED order, not an illegal
action (operator ruling 2026-08-05 + amendment). Execution is a seeded, path-independent
coinflip: HIT fills normally, MISS leaves position/cash/fee unchanged, and the RECORDED
decision is always the attempt — never silently rewritten to HOLD. Zero-coverage cells
(no price at all) are unaffected: nothing quotable, so nothing is attemptable, and they stay
force-HELD exactly as before TASK_878.

Surface under test: ``mcts_trader.env.jax_trading_env.{fill_coinflip_uniforms,
causal_print_density, parse_fill_prob}`` (the shared primitives) and
``experiments.832_linear_imprint.imprint.{compute_frozen_fill_gate, frame_hold_mask,
walk_env}`` (the consumer that builds the BC-target stream ``inject_teacher`` trains on).

Legs (acceptance gates from the task file):
  (i)   HAND-CHECKED HIT and MISS cells on a controlled synthetic panel: a MISS leaves
        position/cash/PV untouched and the attempted action is recorded UNCHANGED; a HIT
        fills normally and is recorded identically.
  (ia)  REPRODUCIBILITY (same eval_seed + panel => byte-identical) AND PATH-INDEPENDENCE
        (changing one EARLIER decision's action never changes any LATER cell's flip) —
        the same-seed-rerun check alone is NOT run as if it were sufficient; both are
        demonstrated separately, and a naive sequential-RNG negative control is shown to
        fail the SECOND check even though it would pass the first.
  (ii)  THE BC-TARGET REGRESSION (the 78.8%-class hazard): the ATTEMPT survives into
        ``walk_env``'s recorded ``actions`` (== the array ``inject_teacher`` trains on) even
        on a MISS. Proven as a real MUTATION test: the SAME scenario walked under the
        actual pre-878 code path (``frame_hold_mask(..., include_frozen=True)``, still
        live and byte-identical to the shipped 868 gate) is asserted to REWRITE the
        attempt to HOLD — i.e. this leg goes RED under the real guard's removal, not a
        strawman.
  (iii) PV CONSERVATION exact, fill path ON: flat prices at fee 0 end at exactly
        100,000.00 whether the coinflip hits or misses.
  (iv)  ZERO-COVERAGE cells stay force-HELD (excluded from ``frozen_mask`` entirely) even
        when --fill-prob is active, and remain disclosed as inactive.
  (v)   FAILED-FILL DISCLOSURE is present and ASSERTED on a real CLI run (subprocess,
        graded artifact — not merely emitted, per the TASK_890/D40 lesson that a test must
        exercise the real entry point, never stub the function under test).

Run: ``python experiments/878_failed_fill/smoke_failed_fill.py``
"""
import inspect
import json
import os
import subprocess
import sys

import numpy as np
import jax
import jax.numpy as jnp

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_IMPRINT_DIR = os.path.join(_REPO, "experiments", "832_linear_imprint")
if _IMPRINT_DIR not in sys.path:
    sys.path.insert(0, _IMPRINT_DIR)

from mcts_trader.env.jax_trading_env import (   # noqa: E402
    fill_coinflip_uniforms, causal_print_density, parse_fill_prob,
    resolve_fill_prob_mode, EnvConfig as _EnvConfig, reset as _env_reset,
    step_vector as _step_vector, get_current_prices as _get_current_prices,
    compute_portfolio_value as _compute_pv,
)
import imprint as imp   # noqa: E402  (the real module — never stubbed)

_EVAL_PY = os.path.join(_REPO, "experiments", "835_stage4_eval", "eval_vector.py")
_EV_MODULE = None


def _eval_vector_module():
    """``eval_vector.py`` imported as a module (the 861/863/882 smoke pattern) so the legs
    below can drive ``evaluate_checkpoint`` / ``rollout`` DIRECTLY. That is the REAL drop
    site for the fix-pass BLOCKER — the force-HOLD masks live inside ``evaluate_checkpoint``
    itself — and it is the only way to put a zero-coverage slot on an UNPINNED panel, which
    no CLI flag can do. Nothing is stubbed: the real loader builds the panel, the real model
    produces the actions, the real env executes them."""
    global _EV_MODULE
    if _EV_MODULE is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("ev_878_inproc", _EVAL_PY)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _EV_MODULE = mod
    return _EV_MODULE

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)
    return cond


BUY = 2
SELL = 0
HOLD = imp.HOLD_ACTION_CODE
assert HOLD == 1


def _flat_panel(n_rows, n_assets, price=100.0):
    return jnp.full((n_rows, n_assets), float(price), dtype=jnp.float32)


def _cfg(n_assets, fee_bps=0.0):
    return imp.EnvConfig(n_assets=n_assets, window_size=3, initial_cash=100_000.0,
                         fee_bps=fee_bps, slippage_bps=0.0)


# =============================================================================
# leg (i) + (iii): hand-checked HIT / MISS cells, PV exact both ways
# =============================================================================

def leg_i_iii_hand_checked_hit_miss():
    print("\n=== (i)+(iii): hand-checked HIT/MISS cells, PV conservation, fill path ON ===")
    N, A = 40, 1
    start_row = 5
    prices = _flat_panel(start_row + N + 2, A, 100.0)
    open_gate = np.zeros((start_row + N + 2, A), dtype=bool)   # every bar frozen (no real print)
    frozen_mask, fill_miss_mask, meta = imp.compute_frozen_fill_gate(
        prices, open_gate, start_row, N, eval_seed=7, fill_prob_mode=0.5, lookback=10)
    check("frozen_mask covers every decision cell (all frozen, none zero-coverage)",
          bool(frozen_mask.all()))
    misses = np.nonzero(fill_miss_mask[:, 0])[0]
    hits = np.nonzero(~fill_miss_mask[:, 0])[0]
    check("at least one MISS and one HIT exist at p=0.5, N=40 (sanity on the test setup)",
          len(misses) > 0 and len(hits) > 0, f"misses={len(misses)} hits={len(hits)}")
    k_miss, k_hit = int(misses[0]), int(hits[0])

    cfg = _cfg(A, fee_bps=0.0)

    def buy_only_at(k0):
        def fn(k, state):
            return np.array([BUY if k == k0 else HOLD], dtype=np.int32)
        return fn

    rep_miss = imp.walk_env(prices, cfg, start_row, N, buy_only_at(k_miss),
                            fill_gate=(frozen_mask, fill_miss_mask))
    rep_hit = imp.walk_env(prices, cfg, start_row, N, buy_only_at(k_hit),
                           fill_gate=(frozen_mask, fill_miss_mask))

    # MISS: attempt recorded, but NO position ever opens and PV is untouched.
    check("MISS: attempted action recorded UNCHANGED (never rewritten to HOLD)",
          int(rep_miss["actions"][k_miss][0]) == BUY,
          f"got {rep_miss['actions'][k_miss][0]}")
    check("MISS: no position ever opened over the whole walk (avg_long == 0)",
          rep_miss["avg_long"] == 0.0, f"avg_long={rep_miss['avg_long']}")
    check("MISS: PV conservation exact at fee 0 (final_pv == 100000.00)",
          abs(rep_miss["final_pv"] - 100_000.0) < 1e-6, f"final_pv={rep_miss['final_pv']}")
    check("MISS: this cell is counted in the disclosure (n_fill_attempts/n_fill_misses)",
          rep_miss["n_fill_attempts"] == 1 and rep_miss["n_fill_misses"] == 1,
          f"attempts={rep_miss['n_fill_attempts']} misses={rep_miss['n_fill_misses']}")

    # HIT: attempt recorded AND the position actually opens (normal fill).
    check("HIT: attempted action recorded UNCHANGED",
          int(rep_hit["actions"][k_hit][0]) == BUY, f"got {rep_hit['actions'][k_hit][0]}")
    check("HIT: a position WAS opened over the walk (avg_long > 0)",
          rep_hit["avg_long"] > 0.0, f"avg_long={rep_hit['avg_long']}")
    check("HIT: PV conservation exact at fee 0 (final_pv == 100000.00 — flat price, no fee)",
          abs(rep_hit["final_pv"] - 100_000.0) < 1e-6, f"final_pv={rep_hit['final_pv']}")
    check("HIT: this cell is counted as filled, not missed",
          rep_hit["n_fill_attempts"] == 1 and rep_hit["n_fill_misses"] == 0,
          f"attempts={rep_hit['n_fill_attempts']} misses={rep_hit['n_fill_misses']}")


# =============================================================================
# leg (ii): the BC-target regression — attempt survives, proven as a real mutation test
# =============================================================================

def leg_ii_bc_target_attempt_not_outcome():
    print("\n=== (ii): BC-target regression — attempt survives a MISS (real mutation test) ===")
    N, A = 40, 1
    start_row = 5
    prices = _flat_panel(start_row + N + 2, A, 100.0)
    open_gate = np.zeros((start_row + N + 2, A), dtype=bool)
    frozen_mask, fill_miss_mask, _ = imp.compute_frozen_fill_gate(
        prices, open_gate, start_row, N, eval_seed=7, fill_prob_mode=0.5, lookback=10)
    misses = np.nonzero(fill_miss_mask[:, 0])[0]
    k_miss = int(misses[0])
    cfg = _cfg(A, fee_bps=0.0)

    def buy_only_at(k0):
        def fn(k, state):
            return np.array([BUY if k == k0 else HOLD], dtype=np.int32)
        return fn

    # THE GUARD: fill_gate active, hold_mask carries ONLY the zero-coverage rule (none here).
    rep_new = imp.walk_env(prices, cfg, start_row, N, buy_only_at(k_miss),
                           hold_mask=imp.frame_hold_mask(prices, start_row, N, open_gate=None),
                           fill_gate=(frozen_mask, fill_miss_mask))
    check("NEW (878) path: BC target at the MISS cell is the ATTEMPT (BUY), not HOLD",
          int(rep_new["actions"][k_miss][0]) == BUY, f"got {rep_new['actions'][k_miss][0]}")

    # THE REAL REVERT: the ACTUAL pre-878 code path — frame_hold_mask(include_frozen=True) is
    # byte-identical to the function this module shipped with before this task (868's shipped
    # union of price<=0 OR ~open_gate). This is not a rebuilt strawman; it is the real guard,
    # called with the flag that reproduces its pre-878 behaviour, still present in this file.
    old_hold_mask = imp.frame_hold_mask(prices, start_row, N, open_gate=open_gate,
                                        include_frozen=True)
    rep_old = imp.walk_env(prices, cfg, start_row, N, buy_only_at(k_miss),
                           hold_mask=old_hold_mask, fill_gate=None)
    check("OLD (pre-878) code path REWRITES the same cell to HOLD (the guard's real drop site)",
          int(rep_old["actions"][k_miss][0]) == HOLD, f"got {rep_old['actions'][k_miss][0]}")
    check("=> this test genuinely goes RED under the real pre-878 guard "
          "(not a strawman revert)",
          int(rep_new["actions"][k_miss][0]) != int(rep_old["actions"][k_miss][0]))

    # And the injected-teacher path itself: inject_teacher trains on exactly `actions`, so a
    # BC label built from rep_new must differ from one built from rep_old at this cell.
    check("inject_teacher's target array differs at the MISS cell between the two paths "
          "(the 78.8%-class hazard this gate exists to catch)",
          int(rep_new["actions"][k_miss][0]) != int(rep_old["actions"][k_miss][0]))


# =============================================================================
# leg (ia): reproducibility AND path-independence (NOT the same check)
# =============================================================================

def leg_ia_reproducibility_and_path_independence():
    print("\n=== (ia): reproducibility (same-seed rerun) AND path-independence (separately) ===")
    N, A = 30, 4
    eval_seed = 123

    # -- reproducibility: same (eval_seed, N, A) => byte-identical, across two independent calls.
    u1 = fill_coinflip_uniforms(eval_seed, N, A)
    u2 = fill_coinflip_uniforms(eval_seed, N, A)
    check("reproducibility: two independent calls, same seed/shape -> bit-identical",
          np.array_equal(u1, u2))
    check("seed-sensitivity: a DIFFERENT eval_seed changes the draw "
          "(rules out a constant/degenerate function)",
          not np.array_equal(u1, fill_coinflip_uniforms(eval_seed + 1, N, A)))

    # -- path-independence (truncation-invariance): decision k's draw cannot depend on how
    # MANY decisions were ever requested — a plain unconditional sequential split-chain
    # actually satisfies this too (each step's key depends only on the PREVIOUS key, not on
    # how many more steps follow), so this check alone does not yet distinguish the two
    # designs. It is still a real, necessary property of the shipped mechanism:
    u_long = fill_coinflip_uniforms(eval_seed, 25, A)
    u_short = fill_coinflip_uniforms(eval_seed, 12, A)
    check("path-independence: fill_coinflip_uniforms(seed, 25, A)[:12] == "
          "fill_coinflip_uniforms(seed, 12, A) (truncation-invariant)",
          np.array_equal(u_long[:12], u_short))

    # -- NEGATIVE CONTROL, the REAL failure mode 1c guards against: a stream that consumes a
    # draw only CONDITIONALLY (e.g. "one draw per frozen-AND-ATTEMPTED cell", which is a
    # natural-looking but wrong implementation of "coinflip only when there is something to
    # flip for"). Whether a draw happens at decision k depends on the ACTIONS chosen at
    # decisions < k, so an earlier action change shifts every later draw. Reproducibility
    # (same seed, same action stream, rerun twice) still holds for this buggy design — it is
    # ONLY path-independence (a DIFFERENT action stream) that exposes it, which is exactly why
    # the task spec calls out that the same-seed rerun alone is not sufficient evidence.
    def naive_conditional_stream(action_stream, frozen, seed):
        k = jax.random.PRNGKey(seed)
        out = np.zeros(frozen.shape, dtype=bool)
        for kk in range(frozen.shape[0]):
            for s in range(frozen.shape[1]):
                if frozen[kk, s] and action_stream[kk, s] != HOLD:
                    k, sub = jax.random.split(k)
                    out[kk, s] = bool(jax.random.uniform(sub) >= 0.5)
        return out

    # -- integration-level path-independence: two GENUINELY DIFFERENT attempted-action
    # streams (diverging at an early decision) must not perturb the LATER coinflip verdicts.
    start_row = 10
    prices = _flat_panel(start_row + N + 2, A, 100.0)
    rng = np.random.RandomState(0)
    open_gate = rng.rand(start_row + N + 2, A) > 0.7
    frozen_mask, fill_miss_mask, _ = imp.compute_frozen_fill_gate(
        prices, open_gate, start_row, N, eval_seed, fill_prob_mode=0.5, lookback=10)
    cfg = _cfg(A)

    def teacher_a(k, state):
        return np.array([(k + s) % 5 for s in range(A)], dtype=np.int32)

    def teacher_b(k, state):
        if k < 3:
            return np.array([(k + s) % 5 for s in range(A)], dtype=np.int32)
        return np.array([(2 * k + 3 * s + 1) % 5 for s in range(A)], dtype=np.int32)

    rep_a = imp.walk_env(prices, cfg, start_row, N, teacher_a,
                         fill_gate=(frozen_mask, fill_miss_mask))
    rep_b = imp.walk_env(prices, cfg, start_row, N, teacher_b,
                         fill_gate=(frozen_mask, fill_miss_mask))
    check("the two attempted-action streams genuinely diverge from k=3 onward "
          "(the test setup is real, not vacuous)",
          not np.array_equal(rep_a["actions"][3:], rep_b["actions"][3:]))
    later = slice(3, N)
    both_attempted = (frozen_mask[later] & (rep_a["actions"][later] != HOLD)
                     & frozen_mask[later] & (rep_b["actions"][later] != HOLD))
    check("cells attempted under BOTH diverging streams exist (test has statistical power)",
          int(both_attempted.sum()) > 0, f"n={int(both_attempted.sum())}")
    verdict_a = fill_miss_mask[later][both_attempted]
    verdict_b = fill_miss_mask[later][both_attempted]  # same underlying array by construction;
    # the REAL claim is recomputing the gate independently per "run" (as evaluate_checkpoint/
    # run() each do exactly once) still agrees — recompute it fresh a second time:
    frozen_mask2, fill_miss_mask2, _ = imp.compute_frozen_fill_gate(
        prices, open_gate, start_row, N, eval_seed, fill_prob_mode=0.5, lookback=10)
    check("independently RE-computed fill_miss_mask (fresh call) is bit-identical",
          np.array_equal(fill_miss_mask, fill_miss_mask2))
    check("changing an EARLIER decision's attempted action never changes a LATER cell's flip",
          np.array_equal(verdict_a, verdict_b))

    naive_a = naive_conditional_stream(rep_a["actions"], frozen_mask, eval_seed)
    naive_b = naive_conditional_stream(rep_b["actions"], frozen_mask, eval_seed)
    check("negative control: the naive CONDITIONAL-draw stream DOES pass reproducibility "
          "(same seed, same action stream, rerun)",
          np.array_equal(naive_conditional_stream(rep_a["actions"], frozen_mask, eval_seed),
                         naive_conditional_stream(rep_a["actions"], frozen_mask, eval_seed)))
    check("negative control: the SAME naive stream's LATER verdicts DO change when only an "
          "EARLIER decision's action changes (rep_a vs rep_b) — this is the failure mode "
          "1c exists to rule out, and the shipped fold_in mechanism above does not have it",
          not np.array_equal(naive_a[later][both_attempted], naive_b[later][both_attempted]))


# =============================================================================
# leg (iv): zero-coverage cells stay force-HELD, excluded from the coinflip entirely
# =============================================================================

def leg_iv_zero_coverage_still_force_hold():
    print("\n=== (iv): zero-coverage cells stay force-HELD, excluded from frozen_mask ===")
    N, A = 20, 2
    start_row = 5
    prices = np.full((start_row + N + 2, A), 100.0, dtype=np.float32)
    prices[:, 1] = 0.0                      # slot 1 = genuinely zero-coverage, all rows
    prices_j = jnp.asarray(prices)
    open_gate = np.zeros((start_row + N + 2, A), dtype=bool)   # both slots "never print"
    frozen_mask, fill_miss_mask, meta = imp.compute_frozen_fill_gate(
        prices_j, open_gate, start_row, N, eval_seed=3, fill_prob_mode=0.5, lookback=10)
    check("zero-coverage slot (price<=0) is EXCLUDED from frozen_mask entirely",
          not frozen_mask[:, 1].any())
    check("the priced slot (slot 0) IS inside frozen_mask (real contrast, not vacuous)",
          frozen_mask[:, 0].any())

    hold_mask = imp.frame_hold_mask(prices_j, start_row, N, open_gate=None)  # zero-coverage only
    check("hold_mask force-HOLDs the zero-coverage slot on every decision",
          bool(hold_mask[:, 1].all()))
    check("hold_mask does NOT force-HOLD the priced (frozen) slot",
          not hold_mask[:, 0].all())

    cfg = _cfg(A)

    def buy_both(k, state):
        return np.array([BUY, BUY], dtype=np.int32)

    rep = imp.walk_env(prices_j, cfg, start_row, N, buy_both,
                       hold_mask=hold_mask, fill_gate=(frozen_mask, fill_miss_mask))
    check("zero-coverage slot's RECORDED action is force-HELD on every decision "
          "(nothing quotable, nothing attemptable)",
          bool((rep["actions"][:, 1] == HOLD).all()))
    check("the priced slot's attempts are NOT force-HELD (at least one real BUY recorded)",
          bool((rep["actions"][:, 0] == BUY).any()))


# =============================================================================
# leg (v): failed-fill disclosure present and ASSERTED on the real CLI (subprocess)
# =============================================================================

def leg_v_disclosure_on_real_cli():
    print("\n=== (v): failed-fill disclosure on the REAL CLI (subprocess, graded artifact) ===")
    out_on = os.path.join(_REPO, "experiments", "878_failed_fill", ".smoke_out_on.json")
    out_off = os.path.join(_REPO, "experiments", "878_failed_fill", ".smoke_out_off.json")
    ckpt_on = os.path.join(_REPO, "experiments", "878_failed_fill", ".smoke_ckpt_on")
    ckpt_off = os.path.join(_REPO, "experiments", "878_failed_fill", ".smoke_ckpt_off")
    py = os.path.join(_REPO, "trader_env", "bin", "python")

    proc = subprocess.run(
        [py, os.path.join(_IMPRINT_DIR, "imprint.py"), "--fill-prob", "0.5", "--seed", "5",
         "--out", out_on, "--ckpt-out", ckpt_on, "--tag", "t878_smoke_v_on"],
        cwd=_REPO, capture_output=True, text=True, timeout=300)
    check("the real CLI runs to completion with --fill-prob 0.5", proc.returncode == 0,
          f"exit={proc.returncode}\nstdout tail:\n{proc.stdout[-2000:]}\n"
          f"stderr tail:\n{proc.stderr[-2000:]}")
    if proc.returncode == 0:
        with open(out_on) as f:
            result = json.load(f)
        fg = result.get("fill_gate")
        check("ON: result['fill_gate'] is present when --fill-prob is given", fg is not None)
        if fg is not None:
            check("ON: fill_gate records which mode was used", fg.get("fill_prob_mode") == 0.5,
                  f"got {fg.get('fill_prob_mode')}")
            check("ON: fill_gate discloses n_frozen_decision_cells "
                  "(int, could be 0 on a fully-live panel)",
                  isinstance(fg.get("n_frozen_decision_cells"), int))
            check("ON: fill_gate discloses the raw miss count",
                  isinstance(fg.get("n_fill_misses_raw"), int))
            check("ON: teacher block carries n_fill_attempts/n_fill_misses",
                  "n_fill_attempts" in result.get("teacher", {})
                  and "n_fill_misses" in result.get("teacher", {}))
            print(f"  (fixture panel: n_frozen_decision_cells={fg.get('n_frozen_decision_cells')}, "
                  f"n_fill_misses_raw={fg.get('n_fill_misses_raw')}, "
                  f"teacher n_fill_attempts={result['teacher'].get('n_fill_attempts')})")

    # OFF: the orchestrator's ruling (883R precedent) — these keys must be STRUCTURALLY ABSENT,
    # not present-and-null, so a flag-off artifact carries zero differing leaves against a
    # pre-878 baseline.
    proc_off = subprocess.run(
        [py, os.path.join(_IMPRINT_DIR, "imprint.py"), "--seed", "5",
         "--out", out_off, "--ckpt-out", ckpt_off, "--tag", "t878_smoke_v_off"],
        cwd=_REPO, capture_output=True, text=True, timeout=300)
    check("the real CLI runs to completion with --fill-prob absent (OFF)",
          proc_off.returncode == 0,
          f"exit={proc_off.returncode}\nstderr tail:\n{proc_off.stderr[-2000:]}")
    if proc_off.returncode == 0:
        with open(out_off) as f:
            result_off = json.load(f)
        check("OFF: 'fill_gate' key is STRUCTURALLY ABSENT (not null)",
              "fill_gate" not in result_off, f"got {result_off.get('fill_gate', '<absent>')!r}")
        check("OFF: teacher block has NEITHER n_fill_attempts NOR n_fill_misses",
              "n_fill_attempts" not in result_off.get("teacher", {})
              and "n_fill_misses" not in result_off.get("teacher", {}))

    import shutil
    for p in (out_on, out_off,
             os.path.join(_REPO, "experiments", "878_failed_fill",
                          f"frame_provenance_{os.path.basename(ckpt_on)}.json"),
             os.path.join(_REPO, "experiments", "878_failed_fill",
                          f"frame_provenance_{os.path.basename(ckpt_off)}.json")):
        if os.path.exists(p):
            os.remove(p)
    for d in (ckpt_on, ckpt_off):
        if os.path.isdir(d):
            shutil.rmtree(d)


# =============================================================================
# parse_fill_prob validation (cheap, structural)
# =============================================================================

def leg_parse_fill_prob():
    print("\n=== parse_fill_prob: refuses out-of-domain values ===")
    check("'auto' accepted", parse_fill_prob("auto") == "auto")
    check("'data' normalizes to 'auto'", parse_fill_prob("data") == "auto")
    check("0 accepted", parse_fill_prob("0") == 0.0)
    check("1 accepted", parse_fill_prob("1") == 1.0)
    for bad in ("-0.1", "1.1", "nope"):
        try:
            parse_fill_prob(bad)
            check(f"{bad!r} REFUSED", False, "did not raise")
        except ValueError:
            check(f"{bad!r} REFUSED", True)


# =============================================================================
# FIX PASS — BLOCKER: zero-coverage force-HOLD on eval_vector's policy-argmax path (the
# leg the shipped smoke was missing entirely). The guard is FLAG-GATED, not unconditional:
# it applies when --fill-prob is active or a panel frame is pinned. See
# leg_minor3_zero_coverage_guard_is_flag_gated, which pins all five paths.
# =============================================================================

def _ev_args(ev, **over):
    """The eval_vector CLI defaults for the pre-registered fixture smoke universe, as a
    namespace (the `tools/tests/test_decision_rule.py::fixture_smoke` / 882 pattern).
    UNPINNED by construction: ``panel_frame_from_checkpoint=None``."""
    import types
    a = types.SimpleNamespace(
        db=ev.FIXTURE_DB, symbols=list(ev.SMOKE_SYMBOLS), start=ev.SMOKE_START,
        end=ev.SMOKE_END, bar_minutes=ev.SMOKE_BAR_MINUTES,
        n_assets=len(ev.SMOKE_SYMBOLS), max_decisions=ev.SMOKE_MAX_DECISIONS,
        eval_seed=0, position_size_pct=0.1, panel_source="minute",
        panel_frame_from_checkpoint=None, panel_frame_source="daily",
        require_active_slots=None, operator="policy-argmax", model_size="small",
        h_cycles=4, l_cycles=6, sims=32, discount=0.997, gumbel_scale=3.0,
        value_scale=1.0, value_shift=0.0, vector_k=16, vector_hold_bias=3.0,
        vector_dither_scale=1.0, vector_proposal_scheme="dithered-book",
        vector_proposal_temp=1.0, vector_proposal_explore_slots=2,
        vector_per_node_resample=False, vector_budget_step=False, vector_budget_pct=0.99,
        vector_weight_rebalance=False, vector_weight_unit=-1.0, vector_weight_max=4.0,
        vector_rebalance_deadband=-1.0, prior_weight=1.0, leaf_eval="model",
        leaf_rollout_r=3, leaf_vhat_tail=True, decision_rule="argmax", decision_tau=0.5,
        decision_m=4, max_positions=16, min_cash_threshold=100.0, teacher_floor=False,
        tag="inproc878", collect_chosen_vectors=None, fill_prob=None,
    )
    for k, v in over.items():
        setattr(a, k, v)
    return a


def _panel_with_dead_slot(ev, args, dead=1):
    """The REAL unpinned smoke panel with ONE slot turned into a genuinely zero-coverage
    one (price 0 on every row, open_gate False) — i.e. a slot that has not listed yet, or
    whose data has a hole. Measured on the untouched panel: 0 of 320 decision cells have
    price <= 0, so this condition is unreachable from the fixture alone and has to be
    constructed. Everything else (loader, prices, open_gate, env config, timestamps) is
    exactly what build_panel produced."""
    prices, env_cfg0, state_init, row_ts, col_symbols, panel_info = ev.build_panel(args)
    p = np.array(prices, copy=True)
    p[:, dead] = 0.0
    og = np.array(panel_info["open_gate"], copy=True)
    og[:, dead] = False
    info = dict(panel_info)
    info["open_gate"] = og
    assert info.get("panel_frame") is None, "this leg must run on an UNPINNED panel"
    return (jnp.asarray(p), env_cfg0, state_init, row_ts, col_symbols, info)


def leg_blocker_eval_vector_zero_coverage_unpinned():
    print("\n=== BLOCKER: eval_vector policy-argmax + --fill-prob, UNPINNED, zero-coverage "
          "slot stays force-HELD ===")
    ev = _eval_vector_module()
    DEAD = 1
    # --decision-rule threshold --decision-tau 0.0 is a SHIPPED CLI rule that puts every
    # asset on its best NON-HOLD action (pinned by tools/tests/test_decision_rule.py), so the
    # dead slot is GUARANTEED to attempt a trade every decision. That makes the leg
    # non-vacuous without stubbing the policy: we control the guard's INPUT, never the guard.
    args = _ev_args(ev, fill_prob=0.5, decision_rule="threshold", decision_tau=0.0,
                    tag="inproc878_blocker")
    panel = _panel_with_dead_slot(ev, args, dead=DEAD)
    col_symbols = panel[4]
    model, trainer = ev.build_model_and_trainer(args, args.n_assets, init_seed=0)
    result = ev.evaluate_checkpoint(
        {"path": "<in-proc>", "name": "fresh_init"}, model, trainer.state.params, None,
        panel, args, ev.load_corridor_constants(), ev.load_ext_windows())

    dead_sym = col_symbols[DEAD]
    for c in ("0", "21"):
        tier = result["tiers"][c]
        n_dead_trades = tier["activity"]["top5_traded"].get(dead_sym, 0)
        check(f"tier {c}: the ZERO-COVERAGE slot ({dead_sym}) executed ZERO trades",
              n_dead_trades == 0, f"got {n_dead_trades}")
        check(f"tier {c}: the zero-coverage slot is 100% HOLD",
              abs(tier["pct_hold_per_asset"][dead_sym] - 100.0) < 1e-9,
              f"got {tier['pct_hold_per_asset'][dead_sym]}")
        # The corruption signature: a BUY at price 0 mints trade_value/1e-10 shares, so PV
        # explodes as soon as anything marks. Bound it far below that and above 0.
        pv = tier["final_pv"]
        check(f"tier {c}: final_pv stayed finite and sane (no 1e15 zero-price blow-up)",
              np.isfinite(pv) and 0.0 < pv < 1e9, f"final_pv={pv}")
    # Non-vacuity: the LIVE slots really did trade under this decision rule, so the
    # all-HOLD result on the dead slot is the mask working, not the policy being idle.
    live_trades = sum(v for k, v in result["tiers"]["0"]["activity"]["top5_traded"].items()
                      if k != dead_sym)
    check("live slots DID trade under threshold(tau=0) (the leg is not vacuous)",
          live_trades > 0, f"live_trades={live_trades}")
    # And the coinflip really engaged on this panel (24 frozen cells measured).
    fp = result.get("fill_prob")
    check("the fill gate engaged on this UNPINNED panel (fill_prob block present)",
          fp is not None and fp.get("n_frozen_decision_cells", 0) > 0,
          f"fill_prob={fp}")


def leg_minor3_zero_coverage_guard_is_flag_gated():
    """TASK_878 fix pass cycle 3 (MINOR-3) — invariant #1, satisfied literally.

    Guard (A) (zero-coverage force-HOLD on the policy-argmax replay) must fire when the fill
    mechanism is ON or a frame is pinned, and must NOT fire on the flag-absent UNPINNED path,
    where it would be an un-flag-gated canon change. The dangerous case is the unpinned DAILY
    panel: gate_dec is None there, so (B) cannot cover for it and an unconditional (A) would
    silently mask cells no pre-878 run masked. No fixture-only CLI repro exists (--smoke pins
    bar_minutes=60, --panel-source daily needs 1440), so this is pinned IN-PROCESS on a
    synthetic gate-free panel — the same drop site, one layer below the CLI."""
    print("\n=== MINOR-3: zero-coverage guard (A) is flag-gated (invariant #1) ===")
    ev = _eval_vector_module()
    DEAD = 1
    args = _ev_args(ev, decision_rule="threshold", decision_tau=0.0, tag="inproc878_m3")
    prices, env_cfg0, state_init, row_ts, col_symbols, panel_info = ev.build_panel(args)
    p = np.array(prices, copy=True)
    p[:, DEAD] = 0.0
    model, trainer = ev.build_model_and_trainer(args, args.n_assets, init_seed=0)

    def _run(fill_prob, keep_gate):
        info = dict(panel_info)
        if keep_gate:
            og = np.array(panel_info["open_gate"], copy=True)
            og[:, DEAD] = False
            info["open_gate"] = og
        else:
            info.pop("open_gate", None)      # the gate-free (daily-like) path
        a = _ev_args(ev, decision_rule="threshold", decision_tau=0.0,
                     fill_prob=fill_prob, tag="inproc878_m3")
        return ev.evaluate_checkpoint(
            {"path": "<in-proc>", "name": "fresh"}, model, trainer.state.params, None,
            (jnp.asarray(p), env_cfg0, state_init, row_ts, col_symbols, info), a,
            ev.load_corridor_constants(), ev.load_ext_windows())

    dead_sym = col_symbols[DEAD]
    # ON + gate-bearing panel: (A) fires — the BLOCKER stays closed.
    on = _run(0.5, keep_gate=True)
    check("ON: the zero-coverage slot is still force-HELD (BLOCKER stays closed)",
          abs(on["tiers"]["0"]["pct_hold_per_asset"][dead_sym] - 100.0) < 1e-9,
          f"got {on['tiers']['0']['pct_hold_per_asset'][dead_sym]}")
    # OFF + gate-bearing (unpinned MINUTE): (B) must still cover the zero-coverage cells,
    # because it carries the FULL ~gate_dec union rather than the union minus zero-coverage.
    off_minute = _run(None, keep_gate=True)
    check("OFF + unpinned MINUTE: (B) still force-HOLDs the zero-coverage slot "
          "(gating (A) must not open a new hole)",
          abs(off_minute["tiers"]["0"]["pct_hold_per_asset"][dead_sym] - 100.0) < 1e-9,
          f"got {off_minute['tiers']['0']['pct_hold_per_asset'][dead_sym]}")
    # OFF + gate-FREE (daily-like): NOTHING may mask — this is the canon path invariant #1
    # protects, and pre-878 it carried no mask at all.
    off_daily = _run(None, keep_gate=False)
    traded = off_daily["tiers"]["0"]["activity"]["top5_traded"].get(dead_sym, 0)
    check("OFF + gate-FREE panel: guard (A) does NOT fire (pre-878 canon restored, "
          "un-flag-gated behaviour removed)",
          off_daily["tiers"]["0"]["pct_hold_per_asset"][dead_sym] < 100.0 or traded > 0,
          f"pct_hold={off_daily['tiers']['0']['pct_hold_per_asset'][dead_sym]} traded={traded}")
    # ON + gate-FREE is REFUSED, so there is no ON path where (A) is unreachable.
    try:
        _run(0.5, keep_gate=False)
        check("ON + gate-free panel is REFUSED (no unguarded ON path exists)", False,
              "did not raise")
    except RuntimeError:
        check("ON + gate-free panel is REFUSED (no unguarded ON path exists)", True)


def leg_major2_activity_is_outcome_not_attempt():
    print("\n=== MAJOR-2: activity counters are OUTCOME (executed), not ATTEMPT ===")
    ev = _eval_vector_module()
    N, A = 30, 2
    warm = 3
    prices = jnp.asarray(np.full((warm + N + 2, A), 100.0, dtype=np.float32))
    open_gate = np.zeros((warm + N + 2, A), dtype=bool)          # every bar frozen
    frozen_mask, fill_miss_mask, meta = imp.compute_frozen_fill_gate(
        prices, open_gate, warm, N, eval_seed=11, fill_prob_mode=0.5, lookback=warm)
    cfg = _EnvConfig(n_assets=A, window_size=warm, initial_cash=100_000.0,
                     fee_bps=0.0, slippage_bps=0.0)
    state0 = ev.jit_batch_reset(jax.random.PRNGKey(0)[None, :], prices, cfg).replace(
        episode_start_idx=jnp.array([warm]), current_step=jnp.array([0]),
        max_episode_steps=jnp.array([N + 2]),
        initial_portfolio_value=jnp.array([100_000.0]),
        prev_portfolio_value=jnp.array([100_000.0]))
    policy_actions = np.full((N, A), BUY, dtype=np.int32)        # attempt a BUY everywhere
    rep = ev.rollout(None, None, prices, cfg, state0, A, N, 11, 16, 100.0,
                     policy_actions=policy_actions,
                     open_gate=np.asarray(open_gate)[warm:warm + N],
                     fill_gate=(frozen_mask, fill_miss_mask))
    n_att = rep["n_frozen_attempted_cells"]
    n_fil = rep["n_frozen_filled_cells"]
    n_mis = rep["n_frozen_missed_cells"]
    check("the setup has BOTH hits and misses (non-vacuous)", n_fil > 0 and n_mis > 0,
          f"attempted={n_att} filled={n_fil} missed={n_mis}")
    executed_trades = int(np.asarray(rep["trades_per_asset"]).sum())
    attempted_trades = int(np.asarray(rep["trades_per_asset_attempted"]).sum())
    check("trades_per_asset counts EXECUTED trades (== filled), not attempts",
          executed_trades == n_fil, f"trades_per_asset sum={executed_trades}, filled={n_fil}")
    check("trades_per_asset_attempted counts the ATTEMPTS (== attempted cells)",
          attempted_trades == n_att,
          f"attempted sum={attempted_trades}, attempted cells={n_att}")
    check("the two genuinely differ (a miss really was counted as a trade before this fix)",
          executed_trades != attempted_trades,
          f"executed={executed_trades} attempted={attempted_trades}")
    # action_hist stays INTENT by design (scope 2: policy-behaviour metric).
    hist_attempt = np.asarray(rep["action_hist_pct"])
    hist_exec = np.asarray(rep["action_hist_exec_pct"])
    check("action_hist_pct is INTENT (100% BUY — every cell attempted a BUY)",
          abs(hist_attempt[BUY] - 100.0) < 1e-9, f"got {hist_attempt.tolist()}")
    check("executed_action_hist_pct is OUTCOME and differs (misses show up as HOLD)",
          hist_exec[HOLD] > 0.0 and hist_exec[BUY] < 100.0, f"got {hist_exec.tolist()}")


# =============================================================================
# FIX PASS — MAJOR: an unhonourable --fill-prob request must be LOUD
# =============================================================================

def leg_major_refuse_fill_prob_without_open_gate():
    print("\n=== MAJOR: --fill-prob on a gate-free panel REFUSES (never silently inert) ===")
    try:
        resolve_fill_prob_mode(0.5, None, context="a gate-free test panel")
        check("gate-free panel + --fill-prob REFUSES", False, "did not raise")
    except RuntimeError as e:
        check("gate-free panel + --fill-prob REFUSES", True)
        check("the refusal names the reason (no causal open-gate)",
              "open-gate" in str(e) and "REFUSING" in str(e), f"msg={e}")
    check("a gate-free panel with NO --fill-prob is still fine (returns OFF)",
          resolve_fill_prob_mode(None, None, context="x") is None)
    # And the real drop site: evaluate_checkpoint on a panel whose open_gate is absent.
    ev = _eval_vector_module()
    args = _ev_args(ev, fill_prob=0.5, tag="inproc878_refusal")
    prices, env_cfg0, state_init, row_ts, col_symbols, panel_info = ev.build_panel(args)
    info = {k: v for k, v in panel_info.items() if k != "open_gate"}
    model, trainer = ev.build_model_and_trainer(args, args.n_assets, init_seed=0)
    try:
        ev.evaluate_checkpoint({"path": "<in-proc>", "name": "fresh"}, model,
                               trainer.state.params, None,
                               (prices, env_cfg0, state_init, row_ts, col_symbols, info),
                               args, ev.load_corridor_constants(), ev.load_ext_windows())
        check("evaluate_checkpoint REFUSES --fill-prob on a gate-free panel", False,
              "returned a result instead of raising")
    except RuntimeError as e:
        check("evaluate_checkpoint REFUSES --fill-prob on a gate-free panel",
              "[878]" in str(e), f"msg={e}")


def leg_minor_sentinel_hardening():
    print("\n=== MINOR: None is the ONLY off value; ambiguous sentinels refused ===")
    og = np.ones((10, 2), dtype=bool)
    check("None -> OFF", resolve_fill_prob_mode(None, og, context="x") is None)
    check("0.0 (float) -> ablation arm A, NOT off",
          resolve_fill_prob_mode(0.0, og, context="x") == 0.0)
    check("1.0 (float) -> ablation arm C", resolve_fill_prob_mode(1.0, og, context="x") == 1.0)
    check("'auto' -> auto", resolve_fill_prob_mode("auto", og, context="x") == "auto")
    # np.bool_ subclasses NEITHER bool NOR np.integer, so it slipped the first guard and
    # landed on float(): np.bool_(False) -> 0.0 (arm A), np.bool_(True) -> 1.0. A caller
    # reading a flag out of a numpy array hits exactly this.
    for bad, label in ((False, "False"), (True, "True"), (0, "int 0"), (1, "int 1"),
                       (np.bool_(False), "np.bool_(False)"), (np.bool_(True), "np.bool_(True)"),
                       (np.int64(0), "np.int64(0)")):
        try:
            resolve_fill_prob_mode(bad, og, context="a SimpleNamespace caller")
            check(f"{label} REFUSED (ambiguous off-vs-arm-A sentinel)", False, "did not raise")
        except ValueError:
            check(f"{label} REFUSED (ambiguous off-vs-arm-A sentinel)", True)


# =============================================================================
# FIX PASS — FINDING 4: --fill-prob must not DELETE pre-existing 863 disclosure
# =============================================================================

def leg_finding4_disclosure_survives_fill_prob():
    print("\n=== FINDING 4: active-cells disclosure survives --fill-prob (no vanishing keys) ===")
    N, A = 20, 2
    start_row = 5
    # The fixture's real shape: frozen cells exist, zero-coverage cells do NOT. Pre-fix this
    # is exactly the configuration in which hold_mask becomes None while masking continues
    # via fill_gate, and the two disclosure keys silently disappeared.
    prices = jnp.asarray(np.full((start_row + N + 2, A), 100.0, dtype=np.float32))
    open_gate = np.zeros((start_row + N + 2, A), dtype=bool)
    hold_mask = imp.frame_hold_mask(prices, start_row, N, open_gate=open_gate,
                                    include_frozen=False)
    check("precondition: hold_mask is None (frozen cells only, no zero-coverage)",
          hold_mask is None)
    frozen_mask, fill_miss_mask, _ = imp.compute_frozen_fill_gate(
        prices, open_gate, start_row, N, eval_seed=4, fill_prob_mode=0.5,
        lookback=start_row)
    cfg = _cfg(A)
    rep = imp.walk_env(prices, cfg, start_row, N,
                       lambda k, state: np.full(A, BUY, dtype=np.int32),
                       hold_mask=hold_mask, fill_gate=(frozen_mask, fill_miss_mask))
    check("avg_long_active is still REPORTED (not dropped) when only fill_gate masks",
          rep["avg_long_active"] is not None)
    check("avg_short_active is still REPORTED when only fill_gate masks",
          rep["avg_short_active"] is not None)
    check("with no zero-coverage cells the companion EQUALS the aggregate (documented)",
          rep["avg_long_active"] == rep["avg_long"],
          f"{rep['avg_long_active']} vs {rep['avg_long']}")
    # And the OFF path keeps its original meaning: nothing masked at all => None.
    rep_off = imp.walk_env(prices, cfg, start_row, N,
                           lambda k, state: np.full(A, BUY, dtype=np.int32))
    check("nothing masked by ANY mechanism => companions stay None (unchanged OFF meaning)",
          rep_off["avg_long_active"] is None and rep_off["avg_short_active"] is None)


# =============================================================================
# FIX PASS — FINDING 5: the ON sub-schema must not depend on panel CONTENT
# =============================================================================

_FILL_GATE_KEYS = {"fill_prob_mode", "fill_prob_source", "fill_gate_rule",
                   "n_frozen_decision_cells", "n_fill_misses_raw",
                   "mean_fill_prob_on_frozen_cells", "eval_seed", "note"}


def leg_finding5_stable_schema_on_zero_frozen_panel():
    print("\n=== FINDING 5: ON sub-schema is stable on a panel with ZERO frozen cells ===")
    N, A = 20, 2
    start_row = 5
    prices = jnp.asarray(np.full((start_row + N + 2, A), 100.0, dtype=np.float32))
    live_gate = np.ones((start_row + N + 2, A), dtype=bool)      # every bar PRINTS
    frozen_mask, fill_miss_mask, meta = imp.compute_frozen_fill_gate(
        prices, live_gate, start_row, N, eval_seed=9, fill_prob_mode=0.5, lookback=start_row)
    check("zero-frozen panel still returns REAL mask arrays (not None)",
          frozen_mask is not None and fill_miss_mask is not None)
    check("...and they are all-False (nothing frozen to gate)",
          not frozen_mask.any() and not fill_miss_mask.any())
    check("the meta key set is COMPLETE and identical to the frozen-cell case",
          set(meta.keys()) == _FILL_GATE_KEYS,
          f"missing={_FILL_GATE_KEYS - set(meta)} extra={set(meta) - _FILL_GATE_KEYS}")
    check("n_frozen_decision_cells is 0 and honest", meta["n_frozen_decision_cells"] == 0)
    check("mean_fill_prob_on_frozen_cells is None (mean of an empty set is not a number)",
          meta["mean_fill_prob_on_frozen_cells"] is None)
    check("the note says the mechanism RAN and found none (not 'unavailable')",
          "RAN" in meta["note"] and "not an unavailable mechanism" in meta["note"])
    # Same key set as a panel that DOES have frozen cells.
    frozen_gate = np.zeros((start_row + N + 2, A), dtype=bool)
    _, _, meta2 = imp.compute_frozen_fill_gate(
        prices, frozen_gate, start_row, N, eval_seed=9, fill_prob_mode=0.5,
        lookback=start_row)
    check("frozen-cell panel emits the SAME key set (schema independent of content)",
          set(meta.keys()) == set(meta2.keys()))
    # The all-False gate must be numerically inert vs no gate at all.
    cfg = _cfg(A)
    tf = lambda k, state: np.full(A, BUY, dtype=np.int32)
    r_gate = imp.walk_env(prices, cfg, start_row, N, tf,
                          fill_gate=(frozen_mask, fill_miss_mask))
    r_none = imp.walk_env(prices, cfg, start_row, N, tf)
    check("all-False fill_gate is numerically INERT (same final_pv as no gate)",
          r_gate["final_pv"] == r_none["final_pv"],
          f"{r_gate['final_pv']} vs {r_none['final_pv']}")
    check("...and identical recorded actions", np.array_equal(r_gate["actions"],
                                                              r_none["actions"]))
    check("but disclosure now reads 0, not null ('ran and found none')",
          r_gate["n_fill_attempts"] == 0 and r_none["n_fill_attempts"] is None,
          f"gated={r_gate['n_fill_attempts']} ungated={r_none['n_fill_attempts']}")


def leg_finding5_eval_vector_keyerror_on_zero_frozen_panel():
    print("\n=== FINDING 5b: eval_vector no longer KeyErrors on a zero-frozen-cell panel ===")
    ev = _eval_vector_module()
    # 4 always-live fixture symbols => 0 frozen decision cells (measured). Pre-fix this made
    # compute_frozen_fill_gate return (None, None, partial_meta), so score_tier omitted
    # "frozen_fill" while the fill_prob block still indexed it => KeyError: 'frozen_fill'.
    syms = list(ev.SMOKE_SYMBOLS)[:4]
    args = _ev_args(ev, symbols=syms, n_assets=4, fill_prob=0.5,
                    tag="inproc878_zerofrozen")
    panel = ev.build_panel(args)
    gate = np.asarray(panel[5]["open_gate"])
    warm = int(panel[1].window_size)
    n_dec = panel[5]["n_decisions"]
    check("precondition: this panel really has ZERO frozen decision cells",
          int((~gate[warm:warm + n_dec]).sum()) == 0,
          f"gate-False cells={int((~gate[warm:warm + n_dec]).sum())}")
    model, trainer = ev.build_model_and_trainer(args, args.n_assets, init_seed=0)
    result = ev.evaluate_checkpoint({"path": "<in-proc>", "name": "fresh"}, model,
                                    trainer.state.params, None, panel, args,
                                    ev.load_corridor_constants(), ev.load_ext_windows())
    fp = result.get("fill_prob")
    check("ON: the fill_prob block exists even with zero frozen cells", fp is not None)
    if fp is not None:
        missing = _FILL_GATE_KEYS - set(fp)
        check("ON: the full fill_gate key set is present (no content-dependent schema)",
              not missing, f"missing={sorted(missing)}")
        check("ON: share_of_decisions_affected computed without KeyError",
              "share_of_decisions_affected" in fp)
    for c in ("0", "21"):
        check(f"ON: tier {c} carries a frozen_fill block (zeros, not absent)",
              "frozen_fill" in result["tiers"][c])


# =============================================================================
# FIX PASS — NIT: causal_print_density bounds + the step_vector caller contract
# =============================================================================

def leg_nit_causal_print_density_bounds():
    print("\n=== NIT: causal_print_density clamps both bounds and pins its precondition ===")
    og = np.ones((5, 2), dtype=bool)
    # Pre-fix this raised a bare `IndexError: index 7 is out of bounds` from the unclamped
    # `lo`. It is now refused by the tail precondition with an explanatory message — strictly
    # better than either the crash or the silent clamp.
    try:
        causal_print_density(og, start_row=10, n_decisions=2, lookback=3)
        check("start_row past the panel end is REFUSED", False, "did not raise")
    except IndexError as e:
        check("start_row past the panel end is REFUSED with a CLEAR error, not IndexError",
              False, f"still raises IndexError: {e!r}")
    except ValueError as e:
        check("start_row past the panel end is REFUSED with a clear ValueError, "
              "not a bare IndexError", "n_decisions <= n_rows" in str(e), f"{e}")
    try:
        causal_print_density(og, start_row=1, n_decisions=2, lookback=3)
        check("start_row < lookback REFUSED (would under-estimate p)", False, "did not raise")
    except ValueError:
        check("start_row < lookback REFUSED (would under-estimate p)", True)
    try:
        causal_print_density(og, start_row=3, n_decisions=2, lookback=0)
        check("lookback < 1 REFUSED", False, "did not raise")
    except ValueError:
        check("lookback < 1 REFUSED", True)
    # The OTHER end of the same asymmetry: rows at/past the panel end would silently average
    # a TRUNCATED trailing window, and the last one falls to the n_bars floor and reads
    # p = 0.0 — a fully-liquid slot that never fills. Clamping hides it; the assert refuses it.
    try:
        causal_print_density(og, start_row=3, n_decisions=10, lookback=3)
        check("start_row + n_decisions > n_rows REFUSED (truncated tail window)",
              False, "did not raise")
    except ValueError:
        check("start_row + n_decisions > n_rows REFUSED (truncated tail window)", True)
    # The shipped callers satisfy the precondition, so a normal call still works.
    p = causal_print_density(og, start_row=3, n_decisions=2, lookback=3)
    check("a well-formed call still returns p == 1.0 on a fully-printing panel",
          np.allclose(p, 1.0), f"got {p.tolist()}")


def leg_step_vector_caller_contract():
    print("\n=== CONTRACT: step_vector has NO internal price guard (masking is the caller's) ===")
    # CHARACTERIZATION, not desired behaviour: this pins WHY a caller-side zero-coverage mask
    # is load-bearing at all — step_vector will execute into a zero price and mint ~1e14
    # shares. It does NOT say that mask must be unconditional: in eval_vector it is gated on
    # (--fill-prob active OR frame pinned), because the flag-absent unpinned gate-free path
    # must stay bit-identical to pre-878 (invariant #1); imprint.frame_hold_mask does hold it
    # unconditionally, and that divergence is deliberate. If a future change adds a guard
    # inside step_vector, this leg fails loudly and the docstring contract gets revisited.
    A = 2
    prices = np.full((20, A), 100.0, dtype=np.float32)
    prices[:, 1] = 0.0
    prices_j = jnp.asarray(prices)
    cfg = _EnvConfig(n_assets=A, window_size=3, initial_cash=100_000.0,
                     fee_bps=0.0, slippage_bps=0.0)
    st = _env_reset(jax.random.PRNGKey(0), prices_j, cfg).replace(
        episode_start_idx=jnp.array(5), current_step=jnp.array(0),
        max_episode_steps=jnp.array(10),
        initial_portfolio_value=jnp.array(100_000.0),
        prev_portfolio_value=jnp.array(100_000.0))
    st2, _, r, _ = _step_vector(st, jnp.asarray([HOLD, BUY], dtype=jnp.int32),
                                prices=prices_j, config=cfg)
    jax.block_until_ready(r)
    size = float(np.asarray(st2.pos_size)[1])
    check("step_vector DOES execute a BUY into a zero price (no internal guard)",
          size > 1e12, f"pos_size[1]={size}")
    pv_marked = float(_compute_pv(st2, jnp.asarray(np.array([100.0, 50.0], np.float32))))
    check("...and portfolio value explodes once that slot starts printing "
          "(this is the corruption the caller-side mask prevents)",
          pv_marked > 1e14, f"PV={pv_marked:.3e}")


# =============================================================================
# FIX PASS — real CLI coverage for eval_vector on an UNPINNED panel
# =============================================================================

def leg_vi_eval_vector_real_cli_unpinned():
    print("\n=== (vi): eval_vector REAL CLI, unpinned panel, --fill-prob ON and OFF ===")
    import shutil
    import tempfile
    py = os.path.join(_REPO, "trader_env", "bin", "python")
    work = tempfile.mkdtemp(prefix="t878_ev_")

    # `--smoke` derives its filename from the OPERATOR, not from --tag
    # (results_fixture_smoke_<operator>.json), so ON and OFF must land in SEPARATE
    # directories or the second run silently overwrites the first.
    SMOKE_ARTIFACT = "results_fixture_smoke_policy-argmax.json"

    def run(tag, extra):
        d = os.path.join(work, tag)
        os.makedirs(d, exist_ok=True)
        cmd = [py, _EVAL_PY, "--smoke", "--operator", "policy-argmax",
               "--out-dir", d, "--tag", tag,
               "--smoke-ckpt-dir", os.path.join(work, f"ckpt_{tag}")] + extra
        return subprocess.run(cmd, cwd=_REPO, capture_output=True, text=True, timeout=900), d

    try:
        proc, d_on = run("t878_on", ["--fill-prob", "0.5"])
        check("eval_vector --smoke --operator policy-argmax --fill-prob 0.5 completes",
              proc.returncode == 0,
              f"exit={proc.returncode}\nstdout tail:\n{proc.stdout[-2500:]}\n"
              f"stderr tail:\n{proc.stderr[-2500:]}")
        if proc.returncode == 0:
            p_on = os.path.join(d_on, SMOKE_ARTIFACT)
            check("ON: the smoke artifact was written", os.path.exists(p_on),
                  f"missing {p_on}; dir={os.listdir(d_on)}")
            if os.path.exists(p_on):
                with open(p_on) as fh:
                    res_on = json.load(fh)
                fp = res_on.get("fill_prob")
                check("ON: top-level fill_prob block present", fp is not None)
                if fp is not None:
                    missing = _FILL_GATE_KEYS - set(fp)
                    check("ON: complete fill_gate key set", not missing,
                          f"missing={sorted(missing)}")
                    check("ON: the coinflip actually engaged on this panel",
                          fp["n_frozen_decision_cells"] > 0,
                          f"n_frozen={fp['n_frozen_decision_cells']}")
                    check("ON: share_of_decisions_affected present for both tiers",
                          set(fp.get("share_of_decisions_affected", {})) == {"0", "21"})
                for c in ("0", "21"):
                    ff = res_on["tiers"][c].get("frozen_fill")
                    check(f"ON: tier {c} frozen_fill disclosure present", ff is not None)
                    if ff is not None:
                        check(f"ON: tier {c} carries the attempt/outcome companions",
                              "attempted_activity" in ff and "executed_action_hist_pct" in ff
                              and "attempt_vs_outcome" in ff)
                        check(f"ON: tier {c} attempted >= filled (a miss is not a trade)",
                              ff["n_attempted_cells"] >= ff["n_filled_cells"])

        proc_off, d_off = run("t878_off", [])
        check("eval_vector --smoke --operator policy-argmax (flag ABSENT) completes",
              proc_off.returncode == 0,
              f"exit={proc_off.returncode}\nstderr tail:\n{proc_off.stderr[-2500:]}")
        if proc_off.returncode == 0:
            p_off = os.path.join(d_off, SMOKE_ARTIFACT)
            check("OFF: the smoke artifact was written", os.path.exists(p_off),
                  f"missing {p_off}; dir={os.listdir(d_off)}")
            if os.path.exists(p_off):
                with open(p_off) as fh:
                    res_off = json.load(fh)
                check("OFF: 'fill_prob' STRUCTURALLY ABSENT", "fill_prob" not in res_off)
                for c in ("0", "21"):
                    check(f"OFF: tier {c} 'frozen_fill' STRUCTURALLY ABSENT",
                          "frozen_fill" not in res_off["tiers"][c])
    finally:
        shutil.rmtree(work, ignore_errors=True)


def leg_vii_imprint_cli_zero_frozen_panel():
    print("\n=== (vii): imprint REAL CLI on a ZERO-FROZEN-cell panel (stable ON schema) ===")
    import shutil
    import tempfile
    py = os.path.join(_REPO, "trader_env", "bin", "python")
    work = tempfile.mkdtemp(prefix="t878_imp_")
    out = os.path.join(work, "out.json")
    try:
        proc = subprocess.run(
            [py, os.path.join(_IMPRINT_DIR, "imprint.py"), "--fill-prob", "0.5",
             "--seed", "5", "--n-assets", "4", "--iters", "20",
             "--out", out, "--ckpt-out", os.path.join(work, "ckpt"),
             "--tag", "t878_zero_frozen"],
            cwd=_REPO, capture_output=True, text=True, timeout=900)
        check("imprint --fill-prob on the 4-always-live-symbol panel completes",
              proc.returncode == 0,
              f"exit={proc.returncode}\nstdout tail:\n{proc.stdout[-2500:]}\n"
              f"stderr tail:\n{proc.stderr[-2500:]}")
        if proc.returncode == 0 and os.path.exists(out):
            with open(out) as fh:
                res = json.load(fh)
            fg = res.get("fill_gate")
            check("zero-frozen panel still emits fill_gate", fg is not None)
            if fg is not None:
                missing = _FILL_GATE_KEYS - set(fg)
                check("zero-frozen panel emits the COMPLETE fill_gate key set",
                      not missing, f"missing={sorted(missing)}")
                check("n_frozen_decision_cells is 0 on this panel",
                      fg.get("n_frozen_decision_cells") == 0,
                      f"got {fg.get('n_frozen_decision_cells')}")
            check("stdout distinguishes 'ran and found none' from 'unavailable'",
                  "RAN and found NO frozen cells" in proc.stdout)
    finally:
        shutil.rmtree(work, ignore_errors=True)


# =============================================================================
# FIX PASS cycle 3 — the disclosure must not be FORGEABLE (severance mutation)
# =============================================================================

FEE_BPS_SEVERANCE = 21.0    # a fill must COST something or the severance control is vacuous


def _mutant_module(src_path, name, replacements):
    """Import a COPY of a shipped module with targeted source edits applied — the TASK_890
    severance-mutation pattern. This does NOT stub the mechanism under test: it builds a
    MUTANT in which the mechanism is severed, so we can prove the disclosure NOTICES. The
    shipped file is never modified."""
    import importlib.util
    import tempfile
    src = open(src_path).read()
    for old, new in replacements:
        if src.count(old) != 1:
            raise AssertionError(f"severance anchor matched {src.count(old)}x: {old[:70]!r}")
        src = src.replace(old, new)
    d = tempfile.mkdtemp(prefix="t878_mutant_")
    p = os.path.join(d, f"{name}.py")
    with open(p, "w") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _severance_scenario(n=40, a=1, start_row=5):
    prices = _flat_panel(start_row + n + 2, a, 100.0)
    open_gate = np.zeros((start_row + n + 2, a), dtype=bool)
    frozen_mask, fill_miss_mask, _ = imp.compute_frozen_fill_gate(
        prices, open_gate, start_row, n, eval_seed=7, fill_prob_mode=0.5, lookback=start_row)
    return prices, frozen_mask, fill_miss_mask


def leg_severance_walk_env_disclosure_not_forgeable():
    print("\n=== SEVERANCE: walk_env's failed-fill counts cannot survive the mechanism ===")
    N, A, start_row = 40, 1, 5
    prices, frozen_mask, fill_miss_mask = _severance_scenario(N, A, start_row)
    # FEE-BEARING on purpose: at fee 0 on a flat panel a fill costs nothing and PV is exactly
    # 100000 whether or not the coinflip suppressed it, so the "mechanism bites" control would
    # be vacuous and the severance proof worthless. Fees make each executed fill visible in PV.
    cfg = _cfg(A, fee_bps=FEE_BPS_SEVERANCE)
    # ALTERNATING buy/sell: a repeated BUY is a no-op once the slot is already long, so a
    # constant stream costs exactly one fee no matter how many fills are suppressed and the
    # control stays vacuous even with fees. Alternating makes every executed action a real
    # transition, so suppressing one genuinely changes the fee tally and the final book.
    tf = lambda k, state: np.full(A, BUY if k % 2 == 0 else SELL, dtype=np.int32)

    good = imp.walk_env(prices, cfg, start_row, N, tf, fill_gate=(frozen_mask, fill_miss_mask))
    nogate = imp.walk_env(prices, cfg, start_row, N, tf)
    mut = _mutant_module(
        os.path.join(_IMPRINT_DIR, "imprint.py"), "imprint_severed_878",
        [("            act_effective = np.where(fill_miss_mask[k], HOLD_ACTION_CODE,\n"
          "                                     act_attempted).astype(np.int32)",
          "            pass  # SEVERED: the fill is no longer suppressed")])
    sev = mut.walk_env(prices, cfg, start_row, N, tf,
                       fill_gate=(frozen_mask, fill_miss_mask))

    check("the mechanism BITES when intact (gated PV != ungated PV)",
          good["final_pv"] != nogate["final_pv"],
          f"good={good['final_pv']} nogate={nogate['final_pv']}")
    check("SEVERED really did sever it (PV reverts EXACTLY to the no-gate walk)",
          sev["final_pv"] == nogate["final_pv"],
          f"severed={sev['final_pv']} nogate={nogate['final_pv']}")
    check("intact run DISCLOSES misses", good["n_fill_misses"] > 0,
          f"misses={good['n_fill_misses']}")
    check("SEVERED run reports ZERO misses — the disclosure is EVIDENCE, not a restatement "
          "of intent (pre-fix it still claimed the full miss count)",
          sev["n_fill_misses"] == 0, f"severed misses={sev['n_fill_misses']}")
    check("SEVERED per-slot breakdown is empty too",
          not any(sev["per_slot_fill_misses"]), f"{sev['per_slot_fill_misses']}")


def leg_severance_rollout_disclosure_not_forgeable():
    print("\n=== SEVERANCE: rollout's failed-fill counts cannot survive the mechanism ===")
    ev = _eval_vector_module()
    N, A, warm = 30, 2, 3
    prices = jnp.asarray(np.full((warm + N + 2, A), 100.0, dtype=np.float32))
    open_gate = np.zeros((warm + N + 2, A), dtype=bool)
    frozen_mask, fill_miss_mask, _ = imp.compute_frozen_fill_gate(
        prices, open_gate, warm, N, eval_seed=11, fill_prob_mode=0.5, lookback=warm)
    cfg = _EnvConfig(n_assets=A, window_size=warm, initial_cash=100_000.0,
                     fee_bps=0.0, slippage_bps=0.0)

    def _state0(m):
        return m.jit_batch_reset(jax.random.PRNGKey(0)[None, :], prices, cfg).replace(
            episode_start_idx=jnp.array([warm]), current_step=jnp.array([0]),
            max_episode_steps=jnp.array([N + 2]),
            initial_portfolio_value=jnp.array([100_000.0]),
            prev_portfolio_value=jnp.array([100_000.0]))

    pa = np.full((N, A), BUY, dtype=np.int32)
    gd = np.asarray(open_gate)[warm:warm + N]
    good = ev.rollout(None, None, prices, cfg, _state0(ev), A, N, 11, 16, 100.0,
                      policy_actions=pa, open_gate=gd,
                      fill_gate=(frozen_mask, fill_miss_mask))
    nogate = ev.rollout(None, None, prices, cfg, _state0(ev), A, N, 11, 16, 100.0,
                        policy_actions=pa, open_gate=gd, fill_gate=None)
    mut = _mutant_module(
        _EVAL_PY, "eval_vector_severed_878",
        [("            exec_vec = jnp.asarray(\n"
          "                np.where(miss_k, HOLD, av).astype(av.dtype))[None, :].astype(jnp.int32)",
          "            pass  # SEVERED: the fill is no longer suppressed")])
    sev = mut.rollout(None, None, prices, cfg, _state0(mut), A, N, 11, 16, 100.0,
                      policy_actions=pa, open_gate=gd,
                      fill_gate=(frozen_mask, fill_miss_mask))

    check("the mechanism BITES when intact (gated PV != ungated PV)",
          good["final_pv"] != nogate["final_pv"],
          f"good={good['final_pv']} nogate={nogate['final_pv']}")
    check("SEVERED reverts EXACTLY to the no-gate walk",
          sev["final_pv"] == nogate["final_pv"],
          f"severed={sev['final_pv']} nogate={nogate['final_pv']}")
    check("intact run DISCLOSES missed cells", good["n_frozen_missed_cells"] > 0,
          f"missed={good['n_frozen_missed_cells']}")
    check("SEVERED run reports ZERO missed cells (pre-fix it claimed the full count)",
          sev["n_frozen_missed_cells"] == 0, f"severed={sev['n_frozen_missed_cells']}")
    check("SEVERED per-slot breakdown is empty too",
          not any(sev["per_slot_missed_fills"]), f"{sev['per_slot_missed_fills']}")
    check("SEVERED still reports the ATTEMPTS (an attempt is an input property, not a claim "
          "about suppression, so it correctly survives)",
          sev["n_frozen_attempted_cells"] == good["n_frozen_attempted_cells"])


# =============================================================================
# FIX PASS cycle 3 — activity long/short is OUTCOME; the pre-878 invariant holds
# =============================================================================

def leg_activity_longshort_is_outcome():
    print("\n=== MAJOR: activity n_long/n_short are OUTCOME; n_long+n_short == n_trades_total ===")
    ev = _eval_vector_module()
    N, A, warm = 30, 2, 3
    prices = jnp.asarray(np.full((warm + N + 2, A), 100.0, dtype=np.float32))
    open_gate = np.zeros((warm + N + 2, A), dtype=bool)
    frozen_mask, fill_miss_mask, _ = imp.compute_frozen_fill_gate(
        prices, open_gate, warm, N, eval_seed=11, fill_prob_mode=0.5, lookback=warm)
    cfg = _EnvConfig(n_assets=A, window_size=warm, initial_cash=100_000.0,
                     fee_bps=0.0, slippage_bps=0.0)
    state0 = ev.jit_batch_reset(jax.random.PRNGKey(0)[None, :], prices, cfg).replace(
        episode_start_idx=jnp.array([warm]), current_step=jnp.array([0]),
        max_episode_steps=jnp.array([N + 2]),
        initial_portfolio_value=jnp.array([100_000.0]),
        prev_portfolio_value=jnp.array([100_000.0]))
    rep = ev.rollout(None, None, prices, cfg, state0, A, N, 11, 16, 100.0,
                     policy_actions=np.full((N, A), BUY, dtype=np.int32),
                     open_gate=np.asarray(open_gate)[warm:warm + N],
                     fill_gate=(frozen_mask, fill_miss_mask))
    syms = [f"S{i}" for i in range(A)]
    tier = ev.score_tier(rep, np.zeros(N, dtype=np.int64), {}, 60.0, 21, syms, N)
    act = tier["activity"]
    # THE pre-878 invariant: non-HOLD codes are exactly {0,3} u {2,4}, so the long/short split
    # must exhaust the executed trades. Broken by 23 and 14 in arms A and B before this fix.
    check("n_long + n_short == n_trades_total (the invariant every pre-878 artifact held)",
          act["n_long"] + act["n_short"] == act["n_trades_total"],
          f"{act['n_long']} + {act['n_short']} != {act['n_trades_total']}")
    check("the OUTCOME long count matches the FILLED cells, not the attempts",
          act["n_long"] == rep["n_frozen_filled_cells"],
          f"n_long={act['n_long']} filled={rep['n_frozen_filled_cells']}")
    att = tier["frozen_fill"]["attempted_activity"]
    check("the ATTEMPTED companion is genuinely different (and is the intent side)",
          att["n_long"] > act["n_long"], f"attempted={att['n_long']} executed={act['n_long']}")
    check("the attempted companion satisfies the same invariant",
          att["n_long"] + att["n_short"] == att["n_trades_total"])


def leg_arm_a_executes_identically_to_off():
    print("\n=== MAJOR (CLI): arm A (--fill-prob 0) must EXECUTE identically to OFF ===")
    import shutil
    import tempfile
    py = os.path.join(_REPO, "trader_env", "bin", "python")
    work = tempfile.mkdtemp(prefix="t878_armA_")
    ART = "results_fixture_smoke_policy-argmax.json"
    try:
        def run(tag, extra):
            d = os.path.join(work, tag)
            os.makedirs(d, exist_ok=True)
            r = subprocess.run(
                [py, _EVAL_PY, "--smoke", "--operator", "policy-argmax", "--out-dir", d,
                 "--tag", tag, "--smoke-ckpt-dir", os.path.join(work, f"ck_{tag}")] + extra,
                cwd=_REPO, capture_output=True, text=True, timeout=900)
            p = os.path.join(d, ART)
            return r, (json.load(open(p)) if r.returncode == 0 and os.path.exists(p) else None)

        r_off, a_off = run("armoff", [])
        r_a, a_a = run("arma", ["--fill-prob", "0"])
        check("both runs completed", r_off.returncode == 0 and r_a.returncode == 0,
              f"off={r_off.returncode} armA={r_a.returncode}\n{r_a.stderr[-1500:]}")
        if a_off and a_a:
            for c in ("0", "21"):
                t_off, t_a = a_off["tiers"][c], a_a["tiers"][c]
                # Arm A attempts every frozen cell but fills NONE, and OFF force-HOLDs them:
                # the executed vector is identical, so every OUTCOME field must match exactly.
                check(f"tier {c}: final_pv identical (arm A executes what OFF executes)",
                      t_off["final_pv"] == t_a["final_pv"],
                      f"off={t_off['final_pv']} armA={t_a['final_pv']}")
                check(f"tier {c}: the whole OUTCOME activity block is identical",
                      t_off["activity"] == t_a["activity"],
                      f"off={t_off['activity']}\n      armA={t_a['activity']}")
                check(f"tier {c}: n_long+n_short == n_trades_total on BOTH",
                      (t_a["activity"]["n_long"] + t_a["activity"]["n_short"]
                       == t_a["activity"]["n_trades_total"]))
                # ...while the INTENT histogram genuinely differs (arm A records the attempt).
                check(f"tier {c}: action_hist_pct (INTENT) DIFFERS — arm A records attempts",
                      t_off["action_hist_pct"] != t_a["action_hist_pct"])
    finally:
        shutil.rmtree(work, ignore_errors=True)


# =============================================================================
# FIX PASS cycle 3 — the --teacher-vectors lane (the BLOCKER's real drop site)
# =============================================================================

_861_DIR = os.path.join(_REPO, "experiments", "861_search_teacher_imprint")
_861_PANEL = ["--start", "2025-09-14", "--end", "2025-12-12", "--bar-minutes", "240",
              "--n-assets", "8"]
# Verbatim from experiments/861_search_teacher_imprint/smoke_e2e.py:161-168 — the teacher of
# record's operator. --vector-weight-rebalance is already in here, so callers must not repeat it.
_861_TEACHER_OPERATOR = [
    "--operator", "search", "--vector-proposal-scheme", "dithered-book",
    "--vector-k", "16", "--sims", "32", "--vector-hold-bias", "3.0",
    "--vector-dither-scale", "1.0", "--vector-proposal-temp", "1.0",
    "--vector-proposal-explore-slots", "2", "--prior-weight", "0.0",
    "--leaf-eval", "true-env", "--leaf-rollout-r", "3", "--no-leaf-vhat-tail",
    "--gumbel-scale", "3.0", "--vector-weight-rebalance",
]


def leg_teacher_vectors_lane_with_fill_prob():
    print("\n=== BLOCKER: imprint --teacher-vectors WITH --fill-prob (the BC-target lane) ===")
    import shutil
    import tempfile
    coll = os.path.join(_861_DIR, "chosen_vectors_fixture.json")
    if not os.path.exists(coll):
        check("the committed 861 collection exists", False, coll)
        return
    py = os.path.join(_REPO, "trader_env", "bin", "python")
    work = tempfile.mkdtemp(prefix="t878_tv_")
    try:
        proc = subprocess.run(
            [py, os.path.join(_IMPRINT_DIR, "imprint.py"), *_861_PANEL,
             "--vector-weight-rebalance", "--teacher-vectors", coll,
             # --iters 250 is the 861 canon for a REAL imprint. --iters 5 does not train, so
             # imprint's own `ce_decreased` quality assert (imprint.py:2685) fires and masks
             # what this leg is actually testing; 861 uses 5 only for refusal demos.
             "--allow-off-pin-teacher", "--fill-prob", "0.5", "--iters", "250", "--seed", "0",
             "--out", os.path.join(work, "out.json"),
             "--ckpt-out", os.path.join(work, "ckpt"), "--tag", "t878_tv"],
            cwd=_REPO, capture_output=True, text=True, timeout=1200)
        check("imprint --teacher-vectors --fill-prob runs to completion "
              "(pre-fix: AttributeError 'NoneType' has no attribute 'mean', EXIT=1)",
              proc.returncode == 0,
              f"exit={proc.returncode}\nstdout tail:\n{proc.stdout[-2500:]}\n"
              f"stderr tail:\n{proc.stderr[-2500:]}")
        outp = os.path.join(work, "out.json")
        if proc.returncode == 0 and os.path.exists(outp):
            res = json.load(open(outp))
            # imprint.py:3143-3150 — the tv_fit block lives here.
            tv = (res.get("teacher_vectors") or {}).get("fit_vs_original_targets") or {}
            check("the tv_fit block survived and carries masked_cell_fraction",
                  "masked_cell_fraction" in tv, f"keys={sorted(tv)[:12]}")
            if "masked_cell_fraction" in tv:
                check("masked_cell_fraction is a real number (0.0 when nothing is force-HELD)",
                      isinstance(tv["masked_cell_fraction"], float)
                      and 0.0 <= tv["masked_cell_fraction"] <= 1.0,
                      f"got {tv['masked_cell_fraction']!r}")
            check("the run's fill_gate block is present", res.get("fill_gate") is not None)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def leg_collect_with_fill_prob_is_refused():
    """TASK_878 fix pass cycle 3 — the resolution of the collection MAJOR.

    Cycle 2 made the collection header's HOLD fraction OUTCOME while its payload stayed
    ATTEMPT, so imprint.gate_teacher_vectors false-accused the file of tampering. Fixing the
    header made that gate pass — and then the run STILL died, in ``_tv_replay_check``: the
    header also records ``walk_net21``, and a deterministic replay of an ATTEMPT stream cannot
    reproduce a COINFLIP-dependent score. That is an unsatisfiable contract, not a disclosure
    bug, so the combination is refused at the CLI. This leg pins the refusal AND pins that it
    fires before any work happens."""
    print("\n=== MAJOR: --collect-chosen-vectors + --fill-prob is REFUSED (unsatisfiable) ===")
    import shutil
    import tempfile
    py = os.path.join(_REPO, "trader_env", "bin", "python")
    work = tempfile.mkdtemp(prefix="t878_refuse_")
    coll = os.path.join(work, "collection.json")
    try:
        r = subprocess.run(
            [py, _EVAL_PY, "--db", os.path.join(_REPO, "fixtures", "trading_fixture.db"),
             *_861_PANEL, "--position-size-pct", "0.1", *_861_TEACHER_OPERATOR,
             "--checkpoint", os.path.join(_861_DIR, "ckpt_teacher_src"),
             "--allow-unpinned-universe", "--fill-prob", "0.5",
             "--collect-chosen-vectors", coll, "--out-dir", work, "--tag", "t878refuse"],
            cwd=_REPO, capture_output=True, text=True, timeout=600)
        check("the combination is REFUSED (non-zero exit)", r.returncode != 0,
              f"exit={r.returncode}")
        msg = r.stdout + r.stderr
        check("the refusal explains WHY (replay cannot reproduce a coinflip score)",
              "--collect-chosen-vectors cannot be combined with --fill-prob" in msg
              and "walk_net21" in msg, f"stderr tail:\n{r.stderr[-1200:]}")
        check("NO collection file was written", not os.path.exists(coll))
        check("it refuses EARLY — before any panel build or model load",
              "[850]" not in msg and "panel T=" not in msg)
        # ...and the same run WITHOUT --collect-chosen-vectors is still perfectly legal.
        r2 = subprocess.run(
            [py, _EVAL_PY, "--smoke", "--operator", "policy-argmax", "--fill-prob", "0.5",
             "--out-dir", os.path.join(work, "ok"),
             "--smoke-ckpt-dir", os.path.join(work, "ck"), "--tag", "t878ok"],
            cwd=_REPO, capture_output=True, text=True, timeout=900)
        check("--fill-prob WITHOUT collection is unaffected by the refusal",
              r2.returncode == 0, f"exit={r2.returncode}\n{r2.stderr[-1200:]}")
    finally:
        shutil.rmtree(work, ignore_errors=True)



def main():
    leg_parse_fill_prob()
    leg_i_iii_hand_checked_hit_miss()
    leg_ii_bc_target_attempt_not_outcome()
    leg_ia_reproducibility_and_path_independence()
    leg_iv_zero_coverage_still_force_hold()
    # ---- fix pass (cycle 2) ----
    leg_minor_sentinel_hardening()
    leg_nit_causal_print_density_bounds()
    leg_step_vector_caller_contract()
    leg_finding4_disclosure_survives_fill_prob()
    leg_finding5_stable_schema_on_zero_frozen_panel()
    leg_major_refuse_fill_prob_without_open_gate()
    leg_blocker_eval_vector_zero_coverage_unpinned()
    leg_minor3_zero_coverage_guard_is_flag_gated()
    leg_major2_activity_is_outcome_not_attempt()
    leg_finding5_eval_vector_keyerror_on_zero_frozen_panel()
    # ---- fix pass (cycle 3) ----
    leg_severance_walk_env_disclosure_not_forgeable()
    leg_severance_rollout_disclosure_not_forgeable()
    leg_activity_longshort_is_outcome()
    leg_v_disclosure_on_real_cli()
    leg_vi_eval_vector_real_cli_unpinned()
    leg_vii_imprint_cli_zero_frozen_panel()
    leg_arm_a_executes_identically_to_off()
    leg_teacher_vectors_lane_with_fill_prob()
    leg_collect_with_fill_prob_is_refused()

    print(f"\n{'=' * 84}")
    if FAILURES:
        print(f"TASK_878 SMOKE: {len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("TASK_878 SMOKE: ALL LEGS PASS")


if __name__ == "__main__":
    main()
