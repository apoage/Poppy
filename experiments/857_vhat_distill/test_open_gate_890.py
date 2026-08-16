#!/usr/bin/env python3
"""TASK_890 (D40 ruling) — regression test: distill.py APPLIES the TASK_868/874 causal OPEN-GATE
to its counterfactual walk, rather than unpacking-and-discarding it.

Background. ``distill.py``'s CLI was DEAD since TASK_868/874 shipped: ``imp.load_training_panel``
returns FIVE values (``prices_j, logc, env_cfg, open_gate, asset_aliases``) but ``run()`` used to
unpack three, raising ``ValueError: too many values to unpack``. D40 ruled that the repair must
not merely fix the arity — it must route ``open_gate`` through ``imp.frame_hold_mask`` exactly as
``imprint.py:2231`` does, and APPLY the resulting mask (force-HOLD wherever a panel slot printed
no real close inside its bar), because distill builds counterfactual v̂-leaf targets from walks
over this same panel and is the warm-start path — an ungated distill would propagate targets that
encode entries on bars that never printed.

WHY THIS FILE LOOKS THE WAY IT DOES (fix-pass, cycle 2). The first version of this test never
entered ``run()``: it read ``open_gate`` itself, built the mask itself, and handed it to
``collect_pass`` explicitly. That only exercised machinery that was already correct and shared
with ``imprint.py``, so reverting ``run()``'s wiring — the ACTUAL defect — left it 11/11 green.
The real drop site is ``run()``: the five-value unpack, the ``frame_hold_mask`` call, and the
``hold_mask=`` forward into ``build_counterfactual_dataset``. LENS 1 below therefore drives the
REAL CLI, in a subprocess, and grades the artifact it writes. Remove the forward (or restore the
unpack-and-discard) and LENS 1 goes red.

  LENS 1 (PRIMARY — acceptance i/ii/iii): run ``distill.py`` as a subprocess with its own
      ``--out``/``--ckpt-out`` inside a temp dir, then assert on the JSON artifact that the gate
      was APPLIED: counts that ``summarize_gate_application`` derives from the passes' own
      force-HOLD branch (zero unless the branch ran), cross-checked against an open-gate cell
      count this file recomputes independently from the panel. Also asserts the CASCADE is
      recorded — gating a WALKED action diverges the portfolio state downstream, so the raw cell
      count is a floor, not the blast radius.
  LENS 2 (acceptance ii, unit): ``imp.frame_hold_mask`` with the causal open-gate present returns
      a STRICTLY LARGER decision-cell mask than without it (fail-closed by construction).
  LENS 3 (candidate level): ``distill.collect_pass`` handed a mask forced False at two chosen
      cells makes EVERY counterfactual candidate HOLD there, for both policies — and the same
      call with ``hold_mask=None`` does not, so the property is non-vacuous.

ZERO monkeypatching of anything under test (acceptance iv): ``run``, ``build_counterfactual_
dataset``, ``collect_pass``, ``frame_hold_mask`` and ``build_root_candidates`` all execute for
real, on the real fixture panel; LENS 1 does not even share a process with them.

SAFETY: the tracked deliverable ``results_fixture_smoke.json`` is hashed before and after LENS 1
and the run is failed if it moved — the bare CLI invocation would overwrite it, so this file
always redirects ``--out``/``--ckpt-out`` into a temp dir and then proves it did.

CPU-only, uses the fixture DB (default distill.py db). Run:
``python experiments/857_vhat_distill/test_open_gate_890.py`` (exit 0 = PASS).
"""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import jax
import numpy as np
from mcts_trader.jax_agent.sampled_search import create_sampled_vector_search

_DISTILL = os.path.join(_REPO, "experiments", "857_vhat_distill", "distill.py")
_TRACKED_DELIVERABLE = os.path.join(_REPO, "experiments", "857_vhat_distill",
                                    "results_fixture_smoke.json")

failures = []


def _check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        failures.append(f"{name}: {detail}")


def _load_by_path(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_REPO, relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _panel(imp, di):
    """Load the fixture panel with the CLI's own fixture defaults (main() lines 1324-1352)."""
    panel_args = argparse.Namespace(
        db=di.FIXTURE_DB, start="2025-09-14", end="2025-12-12", bar_minutes=240,
        n_assets=8, fee_bps=di.FEE_BPS, slip_bps=di.SLIP_BPS, seed=0, reward_attributable=True,
        vector_weight_rebalance=False, vector_weight_unit=-1.0, vector_weight_max=4.0,
        vector_rebalance_deadband=-1.0, position_size_pct=0.5 / 8,
    )
    return imp.load_training_panel(panel_args)


# =============================================================================
# LENS 1 — the REAL CLI, end to end, graded on its artifact
# =============================================================================
def lens1_real_cli(di, imp, prices_j):
    print("\n-- LENS 1: real CLI (subprocess) -> artifact --------------------------------------")
    before = _sha(_TRACKED_DELIVERABLE)
    with tempfile.TemporaryDirectory(prefix="t890_") as tmp:
        out_json = os.path.join(tmp, "results.json")
        ckpt_out = os.path.join(tmp, "ckpt")
        env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_PYTHON_CLIENT_PREALLOCATE="false")
        # --no-smoke-assert: the smoke gates are a SEPARATE acceptance surface (they are currently
        # failing gate (i) at fixture scale — see the TASK_890 fix-pass report and SPEC.md). This
        # lens grades the OPEN-GATE wiring, and must not be hostage to that noisy instrument.
        # --iters/--max-decisions keep it to ~75s; the fixture's gated decision rows are
        # k in {0,2,5,9,...}, so a 10-decision walk still crosses four of them.
        cmd = [sys.executable, _DISTILL, "--no-smoke-assert", "--iters", "5",
               "--max-decisions", "10", "--held-out", "4",
               "--out", out_json, "--ckpt-out", ckpt_out]
        print("  $ " + " ".join(cmd))
        proc = subprocess.run(cmd, cwd=_REPO, env=env, capture_output=True, text=True)
        _check("CLI run() completes (exit 0)", proc.returncode == 0,
               f"rc={proc.returncode}\nSTDERR tail:\n{proc.stderr[-2000:]}")
        if proc.returncode != 0 or not os.path.exists(out_json):
            _check("CLI wrote the redirected artifact", os.path.exists(out_json), out_json)
            return
        with open(out_json) as fh:
            art = json.load(fh)

    _check("SAFETY: tracked results_fixture_smoke.json untouched by this test",
           _sha(_TRACKED_DELIVERABLE) == before, "the tracked deliverable changed")

    # ---- what the gate SHOULD be, recomputed here from the panel, independently of run() -------
    sr, n_dec = int(art["start_row"]), int(art["n_decisions"])
    prices_j_local, open_gate_local = prices_j
    m_gated = imp.frame_hold_mask(prices_j_local, sr, n_dec, open_gate=open_gate_local)
    m_price = imp.frame_hold_mask(prices_j_local, sr, n_dec, open_gate=None)
    exp_force = 0 if m_gated is None else int(np.asarray(m_gated).sum())
    exp_open = exp_force - (0 if m_price is None else int(np.asarray(m_price).sum()))
    _check("fixture window really does contain open-gated decision cells "
           "(otherwise this lens proves nothing)", exp_open > 0,
           f"expected open-gated cells = {exp_open}")

    # ---- the artifact must report the APPLIED gate ---------------------------------------------
    gr = art.get("open_gate")
    _check("artifact carries the open_gate application block", isinstance(gr, dict),
           f"open_gate={gr!r}")
    if not isinstance(gr, dict):
        return

    _check("(ii/iii) artifact n_force_hold_decision_cells is NON-ZERO — the gate was applied",
           int(art.get("n_force_hold_decision_cells", 0)) > 0,
           f"n_force_hold_decision_cells={art.get('n_force_hold_decision_cells')} "
           f"(0 = run() computed a mask and never forwarded it)")
    _check("(ii/iii) artifact n_open_gated_decision_cells is NON-ZERO",
           int(art.get("n_open_gated_decision_cells", 0)) > 0,
           f"n_open_gated_decision_cells={art.get('n_open_gated_decision_cells')}")
    _check("(ii) applied force-HOLD count == the mask this file recomputed from the panel",
           int(art.get("n_force_hold_decision_cells", -1)) == exp_force,
           f"artifact={art.get('n_force_hold_decision_cells')} expected={exp_force}")
    _check("(ii) applied OPEN-GATE count == the open-gate's own contribution, recomputed here",
           int(art.get("n_open_gated_decision_cells", -1)) == exp_open,
           f"artifact={art.get('n_open_gated_decision_cells')} expected={exp_open}")
    _check("(iii) gate applied at EVERY decision of EVERY pass",
           gr.get("applied_in_every_pass") is True,
           f"applied_in_every_pass={gr.get('applied_in_every_pass')}")

    per_pass = gr.get("per_pass") or []
    _check("all four (policy, cost) passes report their own application evidence",
           len(per_pass) == 4, f"per_pass has {len(per_pass)} entries")
    for p in per_pass:
        tag = f"{p.get('policy')}@{p.get('cost')}bp"
        _check(f"[{tag}] masked every walked decision ({p.get('n_decisions_gate_applied')}/"
               f"{p.get('n_decisions_walked')})",
               p.get("n_decisions_walked", 0) > 0
               and p.get("n_decisions_gate_applied") == p.get("n_decisions_walked"),
               f"{p.get('n_decisions_gate_applied')} of {p.get('n_decisions_walked')}")
        _check(f"[{tag}] applied cell count == the recomputed mask",
               p.get("n_cells_masked_applied") == exp_force,
               f"{p.get('n_cells_masked_applied')} vs {exp_force}")

    # ---- the gate must have actually CHANGED something, and the cascade must be recorded --------
    n_cand = int(gr.get("n_candidate_entries_forced_total", 0))
    n_act = int(gr.get("n_walked_action_entries_forced_total", 0))
    _check("(iii) the gate actually FLIPPED entries to HOLD (candidates and/or walked actions) — "
           "not merely 'a mask was present'", (n_cand + n_act) > 0,
           f"candidate entries forced={n_cand}, walked-action entries forced={n_act}")

    n_differ = int(gr.get("n_decisions_targets_differ_total", 0))
    n_total = int(gr.get("n_decisions_total", 0))
    _check("CASCADE recorded: decisions whose targets differ from the ungated construction",
           n_total > 0 and n_differ > 0 and n_differ <= n_total,
           f"n_decisions_targets_differ_total={n_differ} of {n_total}")
    # A pass that moved a WALKED action must count every decision AFTER it: the portfolio state
    # has diverged, so downstream obs/candidates/returns all differ. Not the divergence decision
    # itself — its targets are emitted before its action is taken.
    for p in per_pass:
        fwd = p.get("first_walk_divergence_decision")
        if fwd is None:
            continue
        expect_min = int(p.get("n_decisions_walked", 0)) - int(fwd) - 1
        _check(f"[{p.get('policy')}@{p.get('cost')}bp] cascade counted from the decision after "
               f"the divergence decision {fwd} onward (>= {expect_min})",
               int(p.get("n_decisions_targets_differ", 0)) >= expect_min,
               f"n_decisions_targets_differ={p.get('n_decisions_targets_differ')} "
               f"expected >= {expect_min}")
    _check("artifact states the raw cell count is a FLOOR, not the blast radius",
           "FLOOR" in str(gr.get("cascade_note", "")),
           f"cascade_note={gr.get('cascade_note')!r}")
    print(f"  ... artifact: force-HOLD {art.get('n_force_hold_decision_cells')} cells "
          f"({art.get('n_open_gated_decision_cells')} open-gated) of {art.get('n_decision_cells')}; "
          f"candidate entries forced={n_cand}, walked-action entries forced={n_act}; "
          f"cascade {n_differ}/{n_total} decisions")


# =============================================================================
# LENS 2 + 3 — the masking machinery itself (in-process, no stubbing)
# =============================================================================
def lens23_machinery(di, imp, prices_j, logc, env_cfg, open_gate):
    print("\n-- LENS 2/3: frame_hold_mask + collect_pass (in-process) ---------------------------")
    T, A = logc.shape
    corridor = imp._load_corridor()
    tgt, pres, warm = corridor.book_target_signs(logc)
    start_row = max(int(warm), int(env_cfg.window_size))
    R = 3
    N = min(16, T - 1 - start_row - R - 1)
    assert N >= 8, f"fixture panel too short for this test (N={N})"

    forced_gate = np.array(open_gate, copy=True)
    gated_k, gated_slot0, gated_slot1 = 2, 0, 3
    forced_gate[start_row + gated_k, gated_slot0] = False
    forced_gate[start_row + 5, gated_slot1] = False

    mask_with_gate = imp.frame_hold_mask(prices_j, start_row, N, open_gate=forced_gate)
    mask_without_gate = imp.frame_hold_mask(prices_j, start_row, N, open_gate=None)
    n_with = 0 if mask_with_gate is None else int(mask_with_gate.sum())
    n_without = 0 if mask_without_gate is None else int(mask_without_gate.sum())
    _check("(ii) frame_hold_mask: gate-present mask is non-None",
           mask_with_gate is not None, "mask was None despite a forced-False cell")
    _check("(ii) frame_hold_mask: gate-present decision-cell count STRICTLY LARGER than absent",
           n_with > n_without, f"n_with={n_with} n_without={n_without}")
    if mask_with_gate is None:
        return
    _check("frame_hold_mask marks exactly the two forced cells (plus any real no-print ones)",
           bool(mask_with_gate[gated_k, gated_slot0]) and bool(mask_with_gate[5, gated_slot1]),
           f"mask[{gated_k},{gated_slot0}]={mask_with_gate[gated_k, gated_slot0]} "
           f"mask[5,{gated_slot1}]={mask_with_gate[5, gated_slot1]}")

    build_args = argparse.Namespace(state_dim=64, h_cycles=1, l_cycles=1, lr=1e-4, batch_size=8,
                                    discount=0.997, entropy_coef=0.01, min_entropy=3.0,
                                    reward_hybrid=True, seed=0)
    model, trainer = imp.build_model_and_trainer(build_args, A)
    params = trainer.state.params
    obs_all = imp.encode_training_obs(prices_j, start_row, N, env_cfg)
    helper = create_sampled_vector_search(
        model, n_assets=A, n_actions_per_asset=di.N_ACTIONS_PER_ASSET, num_samples=8,
        num_simulations=8, max_num_considered_actions=8, discount=0.997, gumbel_scale=3.0,
        hold_bias=3.0, dither_scale=1.0, proposal_scheme="dithered-book",
        leaf_eval="true-env", leaf_rollout_r=R, leaf_vhat_tail=False)
    cfg_walk = env_cfg.replace(fee_bps=0.0, slippage_bps=0.0)
    hold_mask = mask_with_gate

    def _run_pass(policy, mask):
        return di.collect_pass(model, params, helper, prices_j, cfg_walk, obs_all, tgt, corridor,
                               start_row, N, A, 0.997, R, policy,
                               jax.random.PRNGKey(1234), 16, 100.0, hold_mask=mask)

    for policy in ("teacher", "search"):
        gated = _run_pass(policy, hold_mask)
        D = gated["cand"].shape[0]
        _check(f"[{policy}] walk reaches both forced decisions before terminating",
               D > 5, f"D={D} (need > 5)")
        if D <= 5:
            continue
        cand_gated = gated["cand"]                                    # (D, M, A)
        _check(f"[{policy}] GATED: every candidate is HOLD at (k={gated_k}, slot={gated_slot0})",
               bool(np.all(cand_gated[gated_k, :, gated_slot0] == di.HOLD)),
               f"cand[{gated_k},:,{gated_slot0}]={cand_gated[gated_k, :, gated_slot0]}")
        _check(f"[{policy}] GATED: every candidate is HOLD at (k=5, slot={gated_slot1})",
               bool(np.all(cand_gated[5, :, gated_slot1] == di.HOLD)),
               f"cand[5,:,{gated_slot1}]={cand_gated[5, :, gated_slot1]}")
        _check(f"[{policy}] the pass's OWN audit reports the cells it masked",
               gated["gate_audit"]["n_cells_masked_applied"] == int(hold_mask[:D].sum()),
               f"{gated['gate_audit']['n_cells_masked_applied']} vs {int(hold_mask[:D].sum())}")

        # Non-vacuity: hold_mask=None is the parameter's own default — if the property above held
        # here too it would be true regardless of gating, i.e. it would prove nothing.
        ungated = _run_pass(policy, None)
        cand_ungated = ungated["cand"]
        still_1 = bool(np.all(cand_ungated[gated_k, :, gated_slot0] == di.HOLD))
        still_2 = bool(np.all(cand_ungated[5, :, gated_slot1] == di.HOLD))
        _check(f"[{policy}] NON-VACUITY (hold_mask=None): NOT all-HOLD at the forced cells",
               not (still_1 and still_2),
               "every candidate was HOLD even without the gate applied — the property above "
               "would pass whether or not the gate is applied, i.e. it is vacuous")
        _check(f"[{policy}] unmasked pass reports ZERO applied cells",
               ungated["gate_audit"]["n_cells_masked_applied"] == 0
               and ungated["gate_audit"]["n_decisions_gate_applied"] == 0,
               f"audit={ungated['gate_audit']}")


def main():
    di = _load_by_path("distill857_test890", "experiments/857_vhat_distill/distill.py")
    imp = di.load_imprint_module()
    prices_j, logc, env_cfg, open_gate, asset_aliases = _panel(imp, di)

    lens1_real_cli(di, imp, (prices_j, open_gate))
    lens23_machinery(di, imp, prices_j, logc, env_cfg, open_gate)

    print("=" * 84)
    if failures:
        print(f"[test_open_gate_890] FAIL — {len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[test_open_gate_890] PASS — the real CLI applies the causal open-gate to its "
          "counterfactual walk, its artifact reports counts derived from the application itself "
          "(not from the mask), the downstream cascade is recorded, and the candidate-level "
          "force-HOLD is proven non-vacuous against the unmasked call.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
