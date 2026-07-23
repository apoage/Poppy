"""TASK_856 — decision-rule probe on the 848 policy-replay operator.

Two kinds of check (spec scope 3 / acceptance i-ii):

  * PROPERTY tests (pure numpy, fast) on ``_apply_decision_rule`` — the rule semantics,
    independent of any model:
      - threshold: acts on the best non-HOLD action per asset iff its marginal mass > tau;
        tau=1.0 => all HOLD; tau=0.0 => never HOLD; tau>=0.5 => the acted action IS the full
        argmax and is non-HOLD ("argmax-when-confident-else-hold", the tau->1 property).
      - top-m: the M selected assets ALWAYS trade their best non-HOLD action (the collapse-
        probe contract — NOT the full argmax, which would re-collapse to HOLD). M=n_assets =>
        every asset trades (== threshold tau=0); M=0 => all HOLD; 0<M<A => exactly M non-HOLD
        trades/step (top-M by non-HOLD mass). A COLLAPSED-regime case (HOLD is the argmax on
        every asset) is the FIX-1 regression guard: argmax trades 0, top-m trades exactly M.

  * FIXTURE SMOKE (model-level, ~1-2 min) — build a fresh-init model on the fixture, replay
    the three rules, assert (a) the argmax rule is BYTE-IDENTICAL to imprint.policy_argmax_actions
    (the HEAD operator), (b) top-m@M=n_assets forces all assets to trade (== threshold tau=0) and
    top-m@M=4 trades exactly 4/step, and (c) the rules produce DIFFERING books. Heavy; not
    collected by pytest — run via
    ``python tools/tests/test_decision_rule.py --fixture-smoke``.

Run:  JAX_PLATFORMS=cpu pytest tools/tests/test_decision_rule.py -q
      JAX_PLATFORMS=cpu python tools/tests/test_decision_rule.py [--fixture-smoke]
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import importlib.util
import sys
import types

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# eval_vector.py lives under a dir whose name starts with a digit (not a python identifier),
# so import it by path — the same pattern the module itself uses for imprint.py.
_EVAL_PY = os.path.join(_REPO, "experiments", "835_stage4_eval", "eval_vector.py")
_spec = importlib.util.spec_from_file_location("eval_vector856", _EVAL_PY)
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

HOLD = ev.HOLD
A5 = ev.N_ACTIONS_PER_ASSET


def _rand_probs(n, a, seed=0):
    """(n, a, 5) per-asset softmax marginals from fixed random logits."""
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((n, a, A5)).astype(np.float64)
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _collapsed_probs(n, a, seed=0):
    """The ep300 collapse signature: HOLD is the argmax on every asset (dominant mass), but
    each asset still carries nonzero, across-asset-varying non-HOLD mass."""
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((n, a, A5)).astype(np.float64) * 0.5
    logits[:, :, HOLD] += 6.0                       # HOLD dominates => argmax == HOLD everywhere
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _ref_best_nonhold(probs):
    """Independent recompute: (best_nonhold_action, best_nonhold_mass, full_argmax)."""
    masked = probs.copy()
    masked[:, :, HOLD] = -1.0
    return masked.argmax(-1), masked.max(-1), probs.argmax(-1)


# ---------------------------------------------------------------------------
# threshold
# ---------------------------------------------------------------------------
def test_threshold_semantics():
    N, A = 50, 8
    probs = _rand_probs(N, A, seed=1)
    bna, bnm, fam = _ref_best_nonhold(probs)
    for tau in (0.0, 0.2, 0.4, 0.5, 0.9, 1.0):
        out = ev._apply_decision_rule(probs, "threshold", tau, m=0, n_assets=A)
        assert out.shape == (N, A) and out.dtype == np.int32
        acted = bnm > tau
        # exact spec: acted -> best non-HOLD action; else HOLD.
        expect = np.where(acted, bna, HOLD).astype(np.int32)
        assert np.array_equal(out, expect), f"threshold tau={tau} mismatch"
        # acted cells are never HOLD (best non-HOLD action is non-HOLD by construction).
        assert not np.any(out[acted] == HOLD)


def test_threshold_tau_one_all_hold():
    # tau=1.0: no single softmax marginal can be strictly > 1 => all HOLD.
    probs = _rand_probs(40, 6, seed=2)
    out = ev._apply_decision_rule(probs, "threshold", 1.0, m=0, n_assets=6)
    assert np.all(out == HOLD)


def test_threshold_tau_zero_never_hold():
    # tau=0.0: every softmax marginal is > 0 => always act on the best non-HOLD action.
    probs = _rand_probs(40, 6, seed=3)
    out = ev._apply_decision_rule(probs, "threshold", 0.0, m=0, n_assets=6)
    assert not np.any(out == HOLD)


def test_threshold_confident_is_argmax():
    """tau->1 (>=0.5) property: wherever the rule acts, the emitted action equals the FULL
    argmax AND is non-HOLD (a >0.5 mass action is the majority => the overall argmax). Inject
    genuinely-confident (>0.999 mass) non-HOLD cells so the high-tau legs assert on non-empty
    slices, not vacuously."""
    N, A = 60, 8
    probs = _rand_probs(N, A, seed=4)
    probs[0, 0, :] = [0.0002, 0.0002, 0.9992, 0.0002, 0.0002]   # action 2 (Buy) @ 0.9992
    probs[1, 3, :] = [0.9994, 0.0001, 0.0002, 0.0002, 0.0001]   # action 0 (Sell) @ 0.9994
    _bna, bnm, fam = _ref_best_nonhold(probs)
    for tau in (0.5, 0.7, 0.9, 0.999):
        out = ev._apply_decision_rule(probs, "threshold", tau, m=0, n_assets=A)
        acted = out != HOLD
        assert acted.any(), f"tau={tau}: no cell acted (assertion would be vacuous)"
        assert np.array_equal(out[acted], fam[acted]), f"tau={tau}: acted != full argmax"
        assert np.all(bnm[acted] > tau)
        assert not np.any(fam[acted] == HOLD)


# ---------------------------------------------------------------------------
# top-m
# ---------------------------------------------------------------------------
def test_topm_m_equals_nassets_trades_all():
    """M=n_assets forces EVERY asset onto its best non-HOLD action (== threshold tau=0) — the
    selected assets always trade. NOT the argmax rule (which can pick HOLD): the collapse-probe
    contract. M>n_assets clamps to n_assets."""
    N, A = 50, 8
    probs = _rand_probs(N, A, seed=5)
    topm = ev._apply_decision_rule(probs, "top-m", tau=0.0, m=A, n_assets=A)
    thr0 = ev._apply_decision_rule(probs, "threshold", 0.0, m=0, n_assets=A)
    assert np.array_equal(topm, thr0)
    assert not np.any(topm == HOLD)                 # every asset trades
    topm_over = ev._apply_decision_rule(probs, "top-m", tau=0.0, m=A + 5, n_assets=A)
    assert np.array_equal(topm_over, thr0)


def test_topm_m_zero_all_hold():
    probs = _rand_probs(30, 6, seed=6)
    out = ev._apply_decision_rule(probs, "top-m", tau=0.0, m=0, n_assets=6)
    assert np.all(out == HOLD)


def test_topm_selects_top_by_nonhold_mass():
    """0<M<A selects exactly the top-M assets by non-HOLD mass (ties -> lower slot) and forces
    each onto its BEST NON-HOLD action; the rest HOLD => exactly M non-HOLD actions per row."""
    N, A, M = 40, 8, 4
    probs = _rand_probs(N, A, seed=7)
    bna, bnm, _fam = _ref_best_nonhold(probs)
    out = ev._apply_decision_rule(probs, "top-m", tau=0.0, m=M, n_assets=A)
    # independent reference: best non-HOLD action on the top-M assets by non-HOLD mass.
    ref = np.full((N, A), HOLD, dtype=np.int32)
    order = np.argsort(-bnm, axis=1, kind="stable")
    sel = order[:, :M]
    rows = np.arange(N)[:, None]
    ref[rows, sel] = bna[rows, sel]
    assert np.array_equal(out, ref)
    # selected assets ALWAYS trade => exactly M non-HOLD per row.
    assert np.all((out != HOLD).sum(axis=1) == M)


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------
def test_collapsed_regime_rules_extract_trades():
    """FIX-1 regression guard. On a HOLD-argmax-everywhere net (the collapsed ep300 signature)
    the argmax rule trades 0, but top-m MUST trade exactly M/step and threshold(low tau) must
    trade — the whole point of the probe. The OLD full-argmax top-m would collapse to 0 trades
    here (== argmax); this test fails if that bug returns."""
    N, A, M = 30, 8, 4
    probs = _collapsed_probs(N, A, seed=11)
    # precondition: argmax is HOLD on every (decision, asset).
    argmax = ev._apply_decision_rule(probs, "argmax", 0.0, 0, A)
    assert np.all(argmax == HOLD), "collapsed fixture precondition failed (argmax not all-HOLD)"
    # top-m: exactly M genuine non-HOLD trades per step.
    topm = ev._apply_decision_rule(probs, "top-m", tau=0.0, m=M, n_assets=A)
    n_trades = (topm != HOLD).sum(axis=1)
    assert np.all(n_trades == M), f"top-m did not trade M={M}/step on collapse: {np.unique(n_trades)}"
    assert not np.any(topm[topm != HOLD] == HOLD)          # trades are real non-HOLD actions
    # threshold below the smallest non-HOLD mass extracts trades too.
    _bna, bnm, _fam = _ref_best_nonhold(probs)
    thr = ev._apply_decision_rule(probs, "threshold", float(bnm.min()) * 0.5, 0, A)
    assert np.any(thr != HOLD), "threshold extracted no trades on the collapsed net"
    # explicit before/after: the OLD (buggy) full-argmax top-m produced 0 trades on this input.
    old = np.full((N, A), HOLD, dtype=np.int32)
    order = np.argsort(-bnm, axis=1, kind="stable")
    rows = np.arange(N)[:, None]
    old[rows, order[:, :M]] = probs.argmax(-1)[rows, order[:, :M]]
    assert np.all(old == HOLD), "collapse fixture should make the OLD full-argmax top-m vacuous"
    assert int((old != HOLD).sum()) == 0 and int((topm != HOLD).sum()) == N * M


def test_unknown_rule_raises():
    probs = _rand_probs(4, 4, seed=8)
    try:
        ev._apply_decision_rule(probs, "nonsense", 0.5, 4, 4)
    except ValueError:
        return
    raise AssertionError("unknown rule did not raise")


def test_net_dependent_classifier():
    def A(**kw):
        base = dict(operator="search", prior_weight=1.0, leaf_eval="model",
                    leaf_vhat_tail=True, vector_proposal_scheme="dithered-book")
        base.update(kw)
        return types.SimpleNamespace(**base)
    # canon search + policy-argmax replay => net-dependent
    assert ev.operator_net_dependent(A()) is True
    assert ev.operator_net_dependent(A(operator="policy-argmax")) is True
    # learned pieces individually keep it net-dependent under a true-env leaf
    assert ev.operator_net_dependent(
        A(prior_weight=0.0, leaf_eval="true-env", leaf_vhat_tail=True)) is True   # v-hat tail
    assert ev.operator_net_dependent(
        A(prior_weight=0.5, leaf_eval="true-env", leaf_vhat_tail=False)) is True  # policy prior
    assert ev.operator_net_dependent(
        A(prior_weight=0.0, leaf_eval="true-env", leaf_vhat_tail=False,
          vector_proposal_scheme="policy-prior")) is True                         # policy proposal
    # the vacuous config: ZERO learned components => not net-dependent
    assert ev.operator_net_dependent(
        A(prior_weight=0.0, leaf_eval="true-env", leaf_vhat_tail=False,
          vector_proposal_scheme="dithered-book")) is False


# ---------------------------------------------------------------------------
# fixture smoke (model-level; heavy — run explicitly, not under pytest collection)
# ---------------------------------------------------------------------------
def fixture_smoke():
    import jax  # noqa: F401  (import cost isolated to the heavy path)
    args = types.SimpleNamespace(
        db=ev.FIXTURE_DB, symbols=list(ev.SMOKE_SYMBOLS),
        start=ev.SMOKE_START, end=ev.SMOKE_END, bar_minutes=ev.SMOKE_BAR_MINUTES,
        n_assets=len(ev.SMOKE_SYMBOLS), max_decisions=ev.SMOKE_MAX_DECISIONS, eval_seed=0,
        position_size_pct=0.1, panel_source="minute", panel_frame_from_checkpoint=None,
        model_size="small", h_cycles=4, l_cycles=6, sims=32, discount=0.997,
        gumbel_scale=3.0, value_scale=1.0, value_shift=0.0, vector_k=16,
        vector_hold_bias=3.0, vector_dither_scale=1.0,
        vector_proposal_scheme="dithered-book", vector_proposal_temp=1.0,
        vector_proposal_explore_slots=2, vector_budget_step=False, vector_budget_pct=0.99,
        vector_weight_rebalance=False, vector_weight_unit=-1.0, vector_weight_max=4.0,
        vector_rebalance_deadband=-1.0,
    )
    n_assets = args.n_assets
    print("[fixture-smoke] building panel + fresh-init model on the fixture ...")
    panel = ev.build_panel(args)
    prices, env_cfg0 = panel[0], panel[1]
    warm = int(env_cfg0.window_size)
    n_decisions = panel[5]["n_decisions"]
    model, trainer = ev.build_model_and_trainer(args, n_assets, init_seed=0)
    imp = ev.load_imprint_module()
    obs_all = imp.encode_training_obs(prices, warm, n_decisions, env_cfg0)
    params = trainer.state.params

    a_argmax = ev.policy_decision_actions(model, params, obs_all, n_assets, "argmax", 0.5, 4)
    a_thr = ev.policy_decision_actions(model, params, obs_all, n_assets, "threshold", 0.5, 4)
    a_topm = ev.policy_decision_actions(model, params, obs_all, n_assets, "top-m", 0.5, 4)
    a_topfull = ev.policy_decision_actions(model, params, obs_all, n_assets,
                                           "top-m", 0.5, n_assets)
    a_thr0 = ev.policy_decision_actions(model, params, obs_all, n_assets, "threshold", 0.0, 4)

    # (a) argmax rule is BYTE-IDENTICAL to the HEAD operator.
    a_head = imp.policy_argmax_actions(model, params, obs_all, n_assets)
    assert np.array_equal(a_argmax, a_head), "argmax rule diverged from imprint.policy_argmax_actions"
    # (b) top-m @ M=n_assets forces every asset onto its best non-HOLD action == threshold(tau=0)
    #     (NOT the argmax rule — the selected assets always trade).
    assert np.array_equal(a_topfull, a_thr0), "top-m@M=n_assets != threshold(tau=0) on the real net"
    assert not np.any(a_topfull == HOLD), "top-m@M=n_assets left some asset on HOLD"
    # (b') top-m @ M=4 trades exactly 4 assets/step regardless of what the argmax says.
    assert np.all((a_topm != HOLD).sum(axis=1) == 4), "top-m@M=4 did not trade exactly 4/step"
    # shapes
    for name, arr in (("argmax", a_argmax), ("threshold", a_thr), ("top-m", a_topm)):
        assert arr.shape == (n_decisions, n_assets), f"{name} bad shape {arr.shape}"
    # (c) the rules produce DIFFERING books.
    books = {"argmax": a_argmax, "threshold": a_thr, "top-m": a_topm}
    holdpct = {k: 100.0 * float((v == HOLD).mean()) for k, v in books.items()}
    print(f"[fixture-smoke] pct_hold — argmax {holdpct['argmax']:.1f}  "
          f"threshold {holdpct['threshold']:.1f}  top-m {holdpct['top-m']:.1f}")
    differ = (not np.array_equal(a_argmax, a_thr)) or (not np.array_equal(a_argmax, a_topm))
    assert differ, "decision rules produced identical books on the fixture (expected divergence)"
    assert not np.array_equal(a_thr, a_topm) or differ
    print("[fixture-smoke] PASS — argmax==HEAD, top-m@M=n_assets trades all (==threshold tau=0), "
          "top-m@M=4 trades 4/step, rules differ")


if __name__ == "__main__":
    if "--fixture-smoke" in sys.argv:
        fixture_smoke()
    else:
        fns = [g for n, g in sorted(globals().items())
               if n.startswith("test_") and callable(g)]
        for fn in fns:
            fn()
            print(f"PASS {fn.__name__}")
        print(f"\nall {len(fns)} property tests PASS "
              f"(run with --fixture-smoke for the model-level book-divergence check)")
