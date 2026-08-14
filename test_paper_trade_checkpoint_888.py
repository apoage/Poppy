#!/usr/bin/env python3
"""TASK_888 fix pass (B4) — the leaf-set hazard, and the countermeasure.

Three things are proven here, and they are deliberately separate:

1. **What a SHAPE check CANNOT see (narrowed by TASK_889 — see below).**
   `flax.training.checkpoints.restore_checkpoint` finishes with
   `serialization.from_state_dict(target, restored)`, so the returned pytree is CONGRUENT TO
   THE TARGET. A checkpoint leaf the target does not have is therefore invisible to a SHAPE
   check on the leaves both trees share: `loaded_param_shape_defect` compares only `b.keys() &
   a.keys()`, so a leaf-SET difference (as opposed to a leaf-SHAPE difference) can never surface
   there, demonstrated directly on the vector-action vs scalar-action trees below — pinning the
   narrowed claim in `loaded_param_shape_defect`'s own docstring rather than asserting it in
   prose. The previous docstring said "A PARTIAL restore is refused too", which was an
   overstatement, and the `only_model`/`only_ckpt` branches that pretended to check it were
   unreachable.

2. **TASK_889 (D39) closes the INTEGRATED path anyway, by a DIFFERENT mechanism.** Item 1 is
   still true of `loaded_param_shape_defect` in isolation, but `checkpoint_io.load_checkpoint_into`
   no longer reports false success for THIS class: a checkpoint now records its own
   `vector_action` at save time, and `trainer.load_checkpoint` REFUSES when a target's own
   `vector_action` disagrees with it. The load this test used to demonstrate as a "clean,
   successful load with a wrong fingerprint" now refuses instead, leaving `trainer.state`
   untouched.

3. **The countermeasure ALSO stays real: the TARGET should be the canonical architecture.**
   Belt-and-suspenders — `tools/paper_trade_vectorbt.py` was the site that had drifted (hand-
   built `MuZeroNetworks(...)`/`MuZeroTrainer(...)`, never passing `use_vector_action`), and
   that is still asserted against the function's own source.

CPU-only, no GPU, no DB. Run: ``python test_paper_trade_checkpoint_888.py`` (exit 0 = PASS).
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import importlib.util
import inspect
import shutil
import sys
import tempfile

import jax
import jax.numpy as jnp

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from mcts_trader.training.config import TrainingConfig
from mcts_trader.training.setup import setup_model_and_trainer
from mcts_trader.jax_agent import checkpoint_io as cio

failures = []


def _check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        failures.append(f"{name}: {detail}")


def _canonical_trainer(n_assets=2, seed=0, vector_action=True):
    """The CANONICAL build path — the one every TASK_888 load site now uses."""
    cfg = TrainingConfig(name="test_paper_trade_888", model_size="small",
                         vector_action=vector_action, seed=seed)
    _model, trainer = setup_model_and_trainer(cfg, n_assets)
    trainer.init_train_state(jax.random.PRNGKey(seed),
                             jnp.ones((1, 27, n_assets, n_assets)))
    return trainer


def _leaf_names(params):
    return {jax.tree_util.keystr(p)
            for p, _v in jax.tree_util.tree_flatten_with_path(params)[0]}


def test_b4_silent_leaf_drop_is_invisible_at_load_time():
    tmp = tempfile.mkdtemp(prefix="test888_b4_")
    try:
        vec = _canonical_trainer(vector_action=True, seed=0)
        vec_leaves = _leaf_names(vec.state.params)
        _check("B4 setup: the CANONICAL vector-action build carries vec_act_proj "
              "(the leaves tools/paper_trade_vectorbt.py's hand-built model lacked)",
              any("vec_act_proj" in k for k in vec_leaves),
              f"no vec_act_proj leaf among {len(vec_leaves)} leaves")
        ckpt = os.path.join(tmp, "ckpt_vector")
        vec.save_checkpoint(ckpt, overwrite=True)

        scalar = _canonical_trainer(vector_action=False, seed=1)
        scalar_leaves = _leaf_names(scalar.state.params)
        dropped = sorted(vec_leaves - scalar_leaves)
        _check("B4 setup: the drifted (non-vector-action) target really is missing leaves the "
              "checkpoint has", bool(dropped), "the two targets had identical leaf sets")

        # TASK_889 (D39): pre-889 this loaded silently — `evidence["loaded"] is True` with a
        # well-formed but WRONG fingerprint (the claim this test's docstring makes about
        # `loaded_param_shape_defect`, a SHAPE check, specifically). TASK_889 closes the
        # INTEGRATED path with a DIFFERENT mechanism: `vec`'s checkpoint now records
        # vector_action=True at save time, and trainer.load_checkpoint refuses when a target's
        # own vector_action disagrees with it — so `load_checkpoint_into` no longer reports
        # false success for this class, even though the shape check alone still cannot see it
        # (demonstrated directly, below, without relying on a load having happened at all).
        params_before = scalar.state.params
        try:
            cio.load_checkpoint_into(scalar, ckpt)
            _check("TASK_889: a scalar target loading a vector-action checkpoint now REFUSES "
                  "instead of reporting loaded=True with a wrong fingerprint", False,
                  "did not raise")
        except RuntimeError as e:
            _check("TASK_889: ... naming vector_action", "vector_action" in str(e), str(e))
        _check("TASK_889: the refusal leaves scalar.state untouched (never a partial restore)",
              scalar.state.params is params_before, "params were mutated despite the refusal")

        _check(f"B4: loaded_param_shape_defect ITSELF (unchanged by TASK_889) still honestly "
              f"reports NOTHING for the {len(dropped)} checkpoint leaf/leaves "
              f"({', '.join(dropped)[:120]}) that differ between these two trees — it is a "
              f"SHAPE check over the leaves both trees SHARE, and a leaf one tree lacks "
              f"entirely is invisible to it, which is why `only_model`/`only_ckpt` could "
              f"never fire and why TASK_889 needed the save-time architecture record instead",
              cio.loaded_param_shape_defect(vec.state.params, scalar.state.params) is None)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_b4_paper_trade_builds_through_the_canonical_path():
    """SOURCE-LEVEL assertion, stated as such: the defence against the class above is that the
    TARGET is the canonical architecture, so the regression guard is "this site still builds
    through TrainingConfig -> setup_model_and_trainer" rather than a runtime behaviour (running
    the real paper-trading audit needs the vectorbt env and a data source)."""
    path = os.path.join(_REPO, "tools", "paper_trade_vectorbt.py")
    spec = importlib.util.spec_from_file_location("paper_trade_vectorbt_888", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # CODE only — the comment block above the build deliberately NAMES the removed constructors
    # to record what drifted, and a check that cannot tell a comment from a call would forbid
    # documenting the fix.
    src = "\n".join(l for l in inspect.getsource(mod.run_paper_trading).splitlines()
                    if not l.lstrip().startswith("#"))
    _check("B4: paper_trade_vectorbt.run_paper_trading builds through "
          "setup_model_and_trainer (the canonical path)",
          "setup_model_and_trainer(" in src, "canonical builder not called")
    _check("B4: it no longer hand-constructs MuZeroNetworks/MuZeroTrainer (the drift that "
          "dropped vec_act_proj)",
          "MuZeroNetworks(" not in src and "MuZeroTrainer(" not in src,
          "still hand-constructs the model/trainer")
    _check("B4: it loads through the shared fail-closed loader",
          "checkpoint_io.load_checkpoint_into(" in src, "does not use checkpoint_io")
    # fix-pass MINOR 1 — same class as train_jax_fast.py's BLOCKER gate. `--checkpoint` declares
    # no `default=`, so argparse stores None when omitted and '' when passed empty; a truthiness
    # gate collapses the two and an explicitly-requested-but-empty checkpoint silently took the
    # "No checkpoint provided, using random weights" branch with no refusal anywhere. Asserted at
    # source level for the same reason as the checks above: running the real audit needs the
    # vectorbt env and a data source.
    _check("MINOR1: the checkpoint gate tests `is not None`, not truthiness (an explicitly "
          "EMPTY --checkpoint must reach the loader and be refused)",
          "if args.checkpoint is not None:" in src,
          "gate is still `if args.checkpoint:` — an empty value silently means random weights")
    _check("MINOR1: an OMITTED --checkpoint still takes the deliberate random-weights branch",
          "No checkpoint provided, using random weights" in src,
          "the legitimate fresh-init path for this diagnostic tool was removed")


def main():
    print("[test_paper_trade_checkpoint_888] B4: the leaf-set hazard is invisible at load time")
    test_b4_silent_leaf_drop_is_invisible_at_load_time()
    print("[test_paper_trade_checkpoint_888] B4: the site builds through the canonical path")
    test_b4_paper_trade_builds_through_the_canonical_path()
    print("=" * 84)
    if failures:
        print(f"[test_paper_trade_checkpoint_888] FAIL — {len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[test_paper_trade_checkpoint_888] PASS — the narrowed B4 claim is pinned, and the "
          "drifted site now builds canonically.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
