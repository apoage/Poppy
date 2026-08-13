#!/usr/bin/env python3
"""TASK_888 — regression test for real_trade_jax.py's ``load_checkpoint_or_die`` (D38
consequence order item 1: the live-trading site, highest consequence in the queue).

Per the task's own invariant, this NEVER instantiates ``RealTrader`` (it needs a Kraken API
client, a live DB handle, and a portfolio manager that is not even importable in this tree —
``RealPortfolioManager`` is None-guarded) and NEVER touches an exchange. It loads
``real_trade_jax.py`` BY PATH (module exec only — class/function definitions, no
instantiation) and unit-tests ``load_checkpoint_or_die`` directly with a tiny real trainer.

Confirms, independently of the shared checkpoint_io suite: (1) the pre-888 bug — this call
site DISCARDED trainer.load_checkpoint's return value, so trainer.state was NEVER the loaded
checkpoint — is actually fixed (trainer.state is replaced by object identity, not just "no
exception"); (2) a nonexistent/unloadable checkpoint dir calls ``sys.exit(1)`` (the D37 fix,
inlined at the live-trading site with NO fresh-init bypass, unlike the other three sites).

Run: ``python test_real_trade_checkpoint_888.py`` (exit 0 = PASS).
"""
import importlib.util
import os
import shutil
import sys
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import jax
import jax.numpy as jnp

from mcts_trader.training.config import TrainingConfig
from mcts_trader.training.setup import setup_model_and_trainer

failures = []


def _check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        failures.append(f"{name}: {detail}")


def _load_real_trade_module():
    """Module EXEC only — defines load_checkpoint_or_die/RealTrader, instantiates nothing.
    Never construct RealTrader() from this module: it needs a live Kraken API client, a
    trading_data.db handle, and RealPortfolioManager (not importable in this tree) — this test
    verifies the load/refuse logic in isolation, per the task's own instruction, never by
    running the trader."""
    spec = importlib.util.spec_from_file_location(
        "real_trade_jax_test888", os.path.join(_REPO, "real_trade_jax.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_trainer(n_assets=2, seed=0):
    cfg = TrainingConfig(name="test_real_trade_888", model_size="small", vector_action=True,
                         seed=seed)
    model, trainer = setup_model_and_trainer(cfg, n_assets)
    dummy_obs = jnp.ones((1, 27, n_assets, n_assets))
    trainer.init_train_state(jax.random.PRNGKey(seed), dummy_obs)
    return trainer


def test_site_construction_path_keeps_vector_action(rtj):
    """TASK_888 fix pass (the BLOCKER at this row's own site): the model this file builds must
    be the CANONICAL architecture, or a real checkpoint loads into it INCOMPLETELY and says
    nothing.

    WHY THE CHECKS ABOVE CANNOT SEE THIS. ``_tiny_trainer`` builds the same
    ``vector_action=True`` config on BOTH sides of every other check in this file, so the target
    always happens to have every leaf the checkpoint has. The defect is a target that is MISSING
    leaves: ``restore_checkpoint`` ends in ``from_state_dict(target, restored)``, which returns a
    pytree CONGRUENT TO THE TARGET, so a leaf the target lacks is dropped with no error, no
    shape defect (nothing to compare against) and no identity hit (a real load did happen). This
    test therefore builds the target through ``real_trade_jax.build_model_and_trainer`` — the
    SITE's own construction path, which is where the drift lived — and compares against the
    checkpoint's OWN fingerprint rather than merely "it loaded".

    Reproduced before the fix (n_assets=2, tiny): 253 of the checkpoint's 255 leaves restored,
    ``vector_action_embedding.vec_act_proj.{kernel,bias}`` silently dropped, ``loaded=True``,
    evidence fingerprint 7a2ced92552c... instead of the checkpoint's 315c86889deb... . On the
    64-asset production architecture the same shape is 202/204."""
    import numpy as np
    from mcts_trader.jax_agent import checkpoint_io

    tmp = tempfile.mkdtemp(prefix="test888_real_trade_vecact_")
    try:
        # --- a REAL vector-action checkpoint, saved from the site's own builder -------------
        _model_src, trainer_src = rtj.build_model_and_trainer(
            n_assets=2, use_vector_action=True, seed=1)
        src_leaves = {jax.tree_util.keystr(p): np.asarray(v) for p, v in
                      jax.tree_util.tree_flatten_with_path(trainer_src.state.params)[0]}
        vec_leaves = sorted(k for k in src_leaves if "vector_action_embedding" in k)
        _check("the site's builder can build the ACTIVE (vector-action) architecture — it has "
               "vector_action_embedding.vec_act_proj leaves", len(vec_leaves) == 2,
               f"vector-action leaves found: {vec_leaves}")
        ckpt_dir = os.path.join(tmp, "ckpt_vector_action")
        trainer_src.save_checkpoint(ckpt_dir, overwrite=True)
        fp_src = checkpoint_io.param_fingerprint(trainer_src.state.params)

        # --- the site's real construction path must restore it COMPLETELY ------------------
        _model_dst, trainer_dst = rtj.build_model_and_trainer(
            n_assets=2, use_vector_action=True, seed=0)
        evidence = rtj.load_checkpoint_or_die(trainer_dst, ckpt_dir)
        dst_leaves = {jax.tree_util.keystr(p): np.asarray(v) for p, v in
                      jax.tree_util.tree_flatten_with_path(trainer_dst.state.params)[0]}
        _check("BLOCKER: every leaf of a vector-action checkpoint survives the load through the "
               "SITE's construction path (no silently dropped leaves)",
               set(dst_leaves) == set(src_leaves),
               f"missing from target: {sorted(set(src_leaves) - set(dst_leaves))}")
        _check("BLOCKER: the load evidence fingerprint EQUALS the checkpoint's own fingerprint "
               "(a congruent-but-incomplete restore produces a well-formed WRONG one)",
               evidence["param_fingerprint"] == fp_src,
               f"{evidence['param_fingerprint']} != {fp_src}")
        _check("BLOCKER: the vector-action policy projection actually carries the CHECKPOINT's "
               "weights, not this process's random init",
               all(np.allclose(src_leaves[k], dst_leaves[k]) for k in vec_leaves),
               "vec_act_proj values differ from the saved checkpoint")

        # --- negative control: the exact drift this fix closes, still silent by construction -
        # This is what real_trade_jax.py did before the fix (hand-built model, use_vector_action
        # never passed => default False). It must still report a clean, successful load — that
        # is the whole point: NO load-time check can see it, so the only defence is the target
        # architecture, and this control is what makes the assertions above meaningful.
        _model_drift, trainer_drift = rtj.build_model_and_trainer(
            n_assets=2, use_vector_action=False, seed=0)
        evidence_drift = rtj.load_checkpoint_or_die(trainer_drift, ckpt_dir)
        drift_leaves = set(jax.tree_util.keystr(p) for p, _ in
                           jax.tree_util.tree_flatten_with_path(trainer_drift.state.params)[0])
        _check("negative control: a scalar-only target (the pre-fix drift) DROPS exactly the "
               "vector-action leaves", set(vec_leaves) and not (set(vec_leaves) & drift_leaves),
               f"drift target still has {sorted(set(vec_leaves) & drift_leaves)}")
        _check("negative control: and reports loaded=True with a WRONG fingerprint — proving "
               "the evidence line cannot detect this class, only the construction path can",
               evidence_drift["loaded"] is True
               and evidence_drift["param_fingerprint"] != fp_src,
               str(evidence_drift))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    rtj = _load_real_trade_module()
    _check("real_trade_jax.py exposes load_checkpoint_or_die",
          hasattr(rtj, "load_checkpoint_or_die"))
    _check("RealTrader was NOT instantiated by loading this module (this test never runs the "
          "trader)", not hasattr(rtj, "_TEST_INSTANTIATED_MARKER"))

    tmp = tempfile.mkdtemp(prefix="test888_real_trade_")
    try:
        # --- item 1's exact bug: nonexistent checkpoint dir must sys.exit, not print "loaded
        # successfully" while quietly keeping the fresh-init params ------------------------
        trainer = _tiny_trainer(seed=0)
        params_before = trainer.state.params
        bad_path = os.path.join(tmp, "nonexistent_checkpoint")
        exited = {"code": None}
        real_exit = sys.exit
        def _fake_exit(code=0):
            exited["code"] = code
            raise SystemExit(code)
        sys.exit = _fake_exit
        try:
            try:
                rtj.load_checkpoint_or_die(trainer, bad_path)
                raised_or_exited = False
            except SystemExit:
                raised_or_exited = True
        finally:
            sys.exit = real_exit
        _check("nonexistent checkpoint dir calls sys.exit (never returns normally)",
              raised_or_exited and exited["code"] == 1, f"exited={exited}")
        _check("trainer.state.params UNCHANGED after the refused load (never silently "
              "trades on a partially-touched state)",
              trainer.state.params is params_before, "params were mutated despite refusal")

        # --- the ACTUAL item-1 regression: a REAL, loadable checkpoint must end up LIVE on
        # trainer.state (pre-888: the return value was discarded, so this NEVER happened) ---
        ckpt_dir = os.path.join(tmp, "ckpt_real")
        trainer_src = _tiny_trainer(seed=1)
        trainer_src.save_checkpoint(ckpt_dir, overwrite=True)

        trainer2 = _tiny_trainer(seed=0)
        params_before2 = trainer2.state.params
        evidence = rtj.load_checkpoint_or_die(trainer2, ckpt_dir)
        _check("item 1 regression: trainer.state.params is REPLACED by the loaded checkpoint "
              "(object identity — pre-888 this was NEVER true, since the return value was "
              "discarded and self.state was only ever set by init_train_state)",
              trainer2.state.params is not params_before2, "params object identity unchanged")
        _check("load_checkpoint_or_die returns load evidence (resolved_path/step/fingerprint)",
              evidence["loaded"] is True and
              isinstance(evidence["param_fingerprint"], str) and
              len(evidence["param_fingerprint"]) == 64, str(evidence))

        # --- confirm the loaded params actually MATCH the saved ones (not just "different
        # from before" but genuinely the checkpoint's own values) ---------------------------
        import numpy as np
        saved_leaves = jax.tree_util.tree_leaves(trainer_src.state.params)
        loaded_leaves = jax.tree_util.tree_leaves(trainer2.state.params)
        matches = all(np.allclose(np.asarray(a), np.asarray(b))
                      for a, b in zip(saved_leaves, loaded_leaves))
        _check("loaded params are BYTE/VALUE-faithful to what was saved (a real restore, not "
              "an unrelated re-init)", matches, "loaded params != saved params")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    test_site_construction_path_keeps_vector_action(rtj)

    print("=" * 84)
    if failures:
        print(f"[test_real_trade_checkpoint_888] FAIL — {len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[test_real_trade_checkpoint_888] PASS — the live-trading load path refuses "
          "correctly and, on a real checkpoint, actually installs it (item 1 fixed).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
