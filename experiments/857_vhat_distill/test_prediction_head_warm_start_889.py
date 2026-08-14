#!/usr/bin/env python3
"""TASK_889 (D39 scope 5) — regression test for distill.py's warm-start TARGET construction.

Companion to (and deliberately separate from) ``test_warm_start_888.py``, which pins TASK_888's
resolver/refusal wiring in ``resolve_warm_start`` unchanged. This file pins the DIFFERENT defect
TASK_889 closes one layer up, at the call site in ``run()``: ``imp.build_model_and_trainer(args,
A)`` always built ``use_prediction_head=False`` (the 832 imprint recipe), so a ``--warm-start``
checkpoint that DOES carry a prediction head restored into a target congruent to itself minus
that head — ``price_predictor``/``latent_to_market`` leaves silently left at random init
(measured upstream, TASK_888/889: 51 of 253 leaves), no shape defect, no identity miss, a
clean-looking load.

Gate (v): proven by LEAF COUNT — the fixed call site (``_peek_recorded_use_prediction_head`` +
``build_model_and_trainer(..., use_prediction_head=...)``) restores every leaf a prediction-head
checkpoint has, and the evidence fingerprint equals the checkpoint's own. The negative control
shows the OLD call-site pattern (hardcoded False) is no longer merely "wrong" — it is now
CAUGHT by TASK_889's own architecture-binding refusal in ``trainer.load_checkpoint``, so the
851-leaf silent drop this test's docstring describes can no longer reach that call site under
canon plumbing (``imp.build_model_and_trainer`` -> ``checkpoint_io.load_checkpoint_into``).

CPU-only, no GPU, no DB. Run: ``python
experiments/857_vhat_distill/test_prediction_head_warm_start_889.py`` (exit 0 = PASS).
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

import jax
import numpy as np

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


def _args(seed=0):
    return argparse.Namespace(state_dim=64, h_cycles=1, l_cycles=1, lr=1e-4, batch_size=8,
                              discount=0.997, entropy_coef=0.01, min_entropy=3.0,
                              reward_hybrid=True, seed=seed)


def _leaf_dict(params):
    return {jax.tree_util.keystr(p): v for p, v in
            jax.tree_util.tree_flatten_with_path(params)[0]}


def _leaf_names(params):
    return set(_leaf_dict(params))


def main():
    di = _load_by_path("distill857_test889", "experiments/857_vhat_distill/distill.py")
    imp = di.load_imprint_module()
    from mcts_trader.jax_agent import checkpoint_io

    tmp = tempfile.mkdtemp(prefix="test889_distill_predhead_")
    try:
        n_assets = 2
        # --- a REAL prediction-head checkpoint, built through the SAME canonical builder ----
        model_src, trainer_src = imp.build_model_and_trainer(
            _args(seed=1), n_assets, use_prediction_head=True)
        src_leaves = _leaf_names(trainer_src.state.params)
        pred_leaves = sorted(k for k in src_leaves
                             if "price_predictor" in k or "latent_to_market" in k)
        _check("setup: the prediction-head build actually carries price_predictor/"
               "latent_to_market leaves", len(pred_leaves) > 0,
               f"leaves found: {len(src_leaves)} total, none matched")
        ckpt_dir = os.path.join(tmp, "ckpt_pred_head")
        trainer_src.save_checkpoint(ckpt_dir, overwrite=True)
        fp_src = checkpoint_io.param_fingerprint(trainer_src.state.params)

        # --- the peek correctly reads the checkpoint's OWN recorded architecture -----------
        peeked = di._peek_recorded_use_prediction_head(ckpt_dir)
        _check("TASK_889 scope 1/5: the peek reads use_prediction_head=True back from "
               "trainer_metadata.json (save-time recording)", peeked is True, str(peeked))

        # --- gate (v): the FIXED call site restores EVERY leaf, proven by leaf count -------
        model_dst, trainer_dst = imp.build_model_and_trainer(
            _args(seed=0), n_assets, use_prediction_head=peeked)
        args_ws = argparse.Namespace(warm_start=ckpt_dir, allow_fresh_init=False)
        warm_start, checkpoint_load = di.resolve_warm_start(trainer_dst, args_ws)
        dst_leaves = _leaf_names(trainer_dst.state.params)
        _check("gate (v): every leaf of the prediction-head checkpoint survives the warm "
               "start through the FIXED call site (zero dropped leaves)",
               dst_leaves == src_leaves,
               f"missing from target: {sorted(src_leaves - dst_leaves)}; "
               f"extra in target: {sorted(dst_leaves - src_leaves)}")
        _check("gate (v): the load evidence fingerprint EQUALS the checkpoint's own "
               "(a congruent-but-incomplete restore would produce a well-formed WRONG one)",
               checkpoint_load["param_fingerprint"] == fp_src,
               f"{checkpoint_load['param_fingerprint']} != {fp_src}")
        src_by_name = _leaf_dict(trainer_src.state.params)
        dst_by_name = _leaf_dict(trainer_dst.state.params)
        _check("gate (v): the prediction-head leaves actually carry the CHECKPOINT's "
               "weights, not this process's random init",
               all(np.allclose(np.asarray(src_by_name[k]), np.asarray(dst_by_name[k]))
                   for k in pred_leaves),
               "prediction-head values differ from the saved checkpoint")

        # --- negative control: the OLD call-site pattern (hardcoded False) is now CAUGHT --
        # imp.build_model_and_trainer(args, A) with no override reproduces exactly what run()
        # built before this fix. Loading the SAME prediction-head checkpoint into it no longer
        # silently drops leaves — TASK_889's architecture-binding refusal in
        # trainer.load_checkpoint (recorded use_prediction_head=True vs target's False) fires
        # first, through the SAME checkpoint_io.load_checkpoint_into resolve_warm_start uses.
        model_old, trainer_old = imp.build_model_and_trainer(_args(seed=0), n_assets)
        args_ws_old = argparse.Namespace(warm_start=ckpt_dir, allow_fresh_init=False)
        try:
            di.resolve_warm_start(trainer_old, args_ws_old)
            _check("negative control: the OLD call-site pattern (use_prediction_head=False "
                   "hardcode) is now REFUSED, not silently accepted", False,
                   "did not raise — the 51-leaf silent drop is still reachable")
        except RuntimeError as e:
            _check("negative control: ... refusing and naming use_prediction_head",
                   "use_prediction_head" in str(e), str(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("=" * 84)
    if failures:
        print(f"[test_prediction_head_warm_start_889] FAIL — {len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[test_prediction_head_warm_start_889] PASS — distill.py's warm-start target now "
          "matches a prediction-head checkpoint's own recorded architecture (zero dropped "
          "leaves), and the old hardcoded call-site pattern is caught rather than silent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
