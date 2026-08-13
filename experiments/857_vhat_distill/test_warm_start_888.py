#!/usr/bin/env python3
"""TASK_888 — regression test for distill.py's ``resolve_warm_start`` (the fail-closed
warm-start load, D38 scope 2).

Loads distill.py BY PATH (the repo's own ``_load_by_path`` idiom — see trader/selftest.py ST12,
experiments/861_search_teacher_imprint/smoke_e2e.py) and exercises ``resolve_warm_start``
directly with a tiny real trainer, bypassing ``run()``'s panel-loading step entirely.

SCOPE OF THE CLAIM THIS FILE MAKES, precisely (fix-pass M7). For the other three TASK_888
sites, acceptance gate (ii) ("all four sites REFUSE on a nonexistent and on an unloadable
checkpoint, naming the resolved path") is demonstrated through the site's own entry point. For
``distill.py`` it is demonstrated **by direct import only** — this test constructs the module,
calls ``resolve_warm_start``, and asserts the refusal. It does NOT prove the refusal fires from
``python distill.py``.

The reason is a PRE-EXISTING defect at HEAD that TASK_888 neither introduced nor is in scope to
fix: ``imp.load_training_panel(args)`` returns 5 values (``imprint.py:2130`` unpacks 5 correctly
after TASK_868/874 added ``open_gate``/``asset_aliases``) but ``distill.py:669`` still unpacks
3, so ``run()`` raises ``ValueError: too many values to unpack`` BEFORE reaching the warm-start
block. That is a panel-loading arity bug, unrelated to checkpoint loading. It is reported, not
repaired, and it is why this test calls ``resolve_warm_start`` directly instead of shelling out.

Run: ``python experiments/857_vhat_distill/test_warm_start_888.py`` (exit 0 = PASS).
"""
import argparse
import importlib.util
import os
import shutil
import sys
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

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


def _tiny_trainer(imp, n_assets=2, seed=0):
    args = argparse.Namespace(state_dim=64, h_cycles=1, l_cycles=1, lr=1e-4, batch_size=8,
                              discount=0.997, entropy_coef=0.01, min_entropy=3.0,
                              reward_hybrid=True, seed=seed)
    model, trainer = imp.build_model_and_trainer(args, n_assets)
    return trainer


def main():
    di = _load_by_path("distill857_test888", "experiments/857_vhat_distill/distill.py")
    imp = di.load_imprint_module()

    tmp = tempfile.mkdtemp(prefix="test888_857_warmstart_")
    try:
        # --- no --warm-start, no --allow-fresh-init: hard REFUSAL --------------------------
        trainer = _tiny_trainer(imp)
        args = argparse.Namespace(warm_start=None, allow_fresh_init=False)
        try:
            di.resolve_warm_start(trainer, args)
            _check("no --warm-start + no --allow-fresh-init REFUSES", False, "did not raise")
        except RuntimeError as e:
            _check("no --warm-start + no --allow-fresh-init REFUSES", "allow-fresh-init" in str(e),
                  str(e))

        # --- no --warm-start, --allow-fresh-init: explicit fresh-init, no raise -------------
        trainer2 = _tiny_trainer(imp)
        state_before = trainer2.state
        args = argparse.Namespace(warm_start=None, allow_fresh_init=True)
        warm_start, checkpoint_load = di.resolve_warm_start(trainer2, args)
        _check("no --warm-start + --allow-fresh-init proceeds (warm_start is None)",
              warm_start is None, f"warm_start={warm_start!r}")
        _check("no --warm-start + --allow-fresh-init never mutates trainer.state",
              trainer2.state is state_before, "state was replaced")
        _check("no --warm-start + --allow-fresh-init still returns full load evidence",
              checkpoint_load["loaded"] is False and
              isinstance(checkpoint_load["param_fingerprint"], str),
              f"checkpoint_load={checkpoint_load}")

        # --- --warm-start pointing nowhere, no --allow-fresh-init: hard REFUSAL ------------
        trainer3 = _tiny_trainer(imp)
        bad_path = os.path.join(tmp, "nope")
        args = argparse.Namespace(warm_start=bad_path, allow_fresh_init=False)
        try:
            di.resolve_warm_start(trainer3, args)
            _check("bad --warm-start + no --allow-fresh-init REFUSES", False, "did not raise")
        except RuntimeError as e:
            _check("bad --warm-start + no --allow-fresh-init REFUSES, naming the resolved path",
                  os.path.realpath(bad_path) in str(e), str(e))

        # --- --warm-start pointing nowhere, --allow-fresh-init: proceeds, disclosed --------
        trainer4 = _tiny_trainer(imp)
        args = argparse.Namespace(warm_start=bad_path, allow_fresh_init=True)
        warm_start4, cl4 = di.resolve_warm_start(trainer4, args)
        _check("bad --warm-start + --allow-fresh-init proceeds fresh (warm_start is None)",
              warm_start4 is None, f"warm_start={warm_start4!r}")
        _check("bad --warm-start + --allow-fresh-init evidence carries loaded=False",
              cl4["loaded"] is False, str(cl4))

        # --- --warm-start pointing at a REAL checkpoint: loads, returns evidence -----------
        trainer_src = _tiny_trainer(imp)
        ckpt_dir = os.path.join(tmp, "ckpt_real")
        trainer_src.save_checkpoint(ckpt_dir, overwrite=True)
        trainer5 = _tiny_trainer(imp)
        params_before = trainer5.state.params
        args = argparse.Namespace(warm_start=ckpt_dir, allow_fresh_init=False)
        warm_start5, cl5 = di.resolve_warm_start(trainer5, args)
        _check("real --warm-start loads successfully", cl5["loaded"] is True, str(cl5))
        _check("real --warm-start returns a display warm_start string",
              isinstance(warm_start5, str) and len(warm_start5) > 0, f"warm_start={warm_start5!r}")
        _check("real --warm-start actually replaced the fresh-init params",
              trainer5.state.params is not params_before, "params unchanged")
        _check("real --warm-start evidence carries a 64-hex-char fingerprint",
              isinstance(cl5["param_fingerprint"], str) and len(cl5["param_fingerprint"]) == 64,
              str(cl5))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("=" * 84)
    if failures:
        print(f"[test_warm_start_888] FAIL — {len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[test_warm_start_888] PASS — resolve_warm_start green on all 5 states.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
