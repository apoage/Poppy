#!/usr/bin/env python3
"""TASK_888 — regression test for train_jax_fast.py's ``_attempt_checkpoint_load`` (D38
consequence order item 2: "a multi-hour resume silently discards its checkpoint and starts
fresh").

Loads train_jax_fast.py BY PATH (module exec only — ``if __name__ == "__main__"`` guards
``main()``/the ``torch`` import, so this never trains anything) and unit-tests
``_attempt_checkpoint_load`` directly with a tiny real trainer — no CLI args, no data loading,
no GPU (JAX_PLATFORMS is forced to cpu BEFORE jax is imported anywhere in this process).

Run: ``python test_train_jax_fast_checkpoint_888.py`` (exit 0 = PASS).
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

from mcts_trader.jax_agent import checkpoint_io
from mcts_trader.training.cli import parse_args
from mcts_trader.training.config import TrainingConfig
from mcts_trader.training.setup import setup_model_and_trainer

failures = []


def _check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        failures.append(f"{name}: {detail}")


def _load_train_jax_fast_module():
    spec = importlib.util.spec_from_file_location(
        "train_jax_fast_test888", os.path.join(_REPO, "train_jax_fast.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_trainer(n_assets=2, seed=0):
    cfg = TrainingConfig(name="test_train_jax_fast_888", model_size="small", vector_action=True,
                         seed=seed)
    model, trainer = setup_model_and_trainer(cfg, n_assets)
    dummy_obs = jnp.ones((1, 27, n_assets, n_assets))
    trainer.init_train_state(jax.random.PRNGKey(seed), dummy_obs)
    return trainer


def _parse(argv):
    """`config` exactly as `main()` receives it, from the REAL parser — not a hand-built stand-in.

    The whole defect below is about what argparse puts in `load_checkpoint` for a flag that was
    PASSED WITH AN EMPTY VALUE versus one that was never passed, so a fabricated config would
    test the wrong thing."""
    saved = sys.argv[:]
    try:
        sys.argv = ["train_jax_fast.py"] + list(argv)
        return parse_args()
    finally:
        sys.argv = saved


def _gate_predicate(tjf):
    """The gate `main()` ACTUALLY uses, or — if the fix is not in yet — the pre-fix inline
    expression as the baseline, so this test goes RED for the real behavioral reason rather
    than merely "attribute missing".

    `main()` asks ONE named predicate whether a checkpoint load was requested, so a test can
    exercise the same code the run does; two spellings that agree today drifting apart tomorrow
    is the exact failure class TASK_888 exists to close (guard vs loader, one layer up)."""
    fn = getattr(tjf, "_checkpoint_load_requested", None)
    if fn is not None:
        return fn
    return lambda cfg: bool(cfg.load_checkpoint)      # pre-fix: `if config.load_checkpoint:`


def _check_empty_flag_gate(tjf):
    """BLOCKER regression — an EXPLICITLY EMPTY `--load-checkpoint` must reach the loader and be
    REFUSED; an OMITTED one must still skip loading entirely.

    Eight orchestration scripts build this flag programmatically and pass the value
    unconditionally (`tools/run_scheduled_training.py:142`, `tools/orchestration/dojo_v2.py:392`
    and `:470`, `dojo_v3_historical.py:791/:860/:942`, the sweeps). When the shell variable
    holding the checkpoint is unset, what reaches argv is `--load-checkpoint ''` — and argparse
    stores `''`, which is NOT None but IS falsy. A truthiness gate (`if config.load_checkpoint:`)
    therefore collapses an explicitly-passed empty checkpoint into "flag not given": no print, no
    exception, no non-zero exit, and a multi-hour resume proceeds on fresh `init_train_state`
    weights. That failure is invisible in the training curve because a fresh run looks like a
    fresh run — this row's own #1-ranked motivating scenario, surviving at the site it ranks
    first. `is not None` is the correct test because `--load-checkpoint`'s argparse default IS
    `None` (`mcts_trader/training/cli.py:313`), so "omitted" and "given empty" stay
    distinguishable; the same idiom `eval_vector.py` already uses for its collect flags."""
    requested = _gate_predicate(tjf)

    cfg_omitted = _parse([])
    _check("argparse default for --load-checkpoint is None (omitted stays distinguishable "
           "from empty)", cfg_omitted.load_checkpoint is None, repr(cfg_omitted.load_checkpoint))
    _check("OMITTED --load-checkpoint does NOT request a load (unchanged behavior)",
           requested(cfg_omitted) is False or requested(cfg_omitted) == False,
           f"gate said {requested(cfg_omitted)!r} for an omitted flag — would try to load")

    cfg_empty = _parse(["--load-checkpoint", ""])
    _check("an explicitly-passed empty --load-checkpoint parses to '' (not None)",
           cfg_empty.load_checkpoint == "", repr(cfg_empty.load_checkpoint))
    _check("EXPLICITLY EMPTY --load-checkpoint DOES request a load (so it can be refused, "
           "instead of silently resuming on fresh-init weights)",
           bool(requested(cfg_empty)),
           "the gate collapsed an explicitly-passed empty checkpoint into 'flag not given': "
           "training would silently proceed on random-init weights — TASK_888's #1 scenario")

    # ...and once it reaches the loader, it must REFUSE, naming the path it resolved to.
    trainer = _tiny_trainer(seed=0)
    params_before = trainer.state.params
    try:
        tjf._attempt_checkpoint_load(trainer, cfg_empty.load_checkpoint, reset_optimizer=False)
        _check("empty --load-checkpoint REFUSES at the loader", False, "did not raise")
    except checkpoint_io.CheckpointRefusalError as e:
        _check("empty --load-checkpoint REFUSES at the loader, naming the resolved path",
               os.path.realpath("") in str(e), str(e))
    _check("trainer.state.params UNCHANGED after the refused empty-flag load",
           trainer.state.params is params_before, "params were mutated despite refusal")


def main():
    tjf = _load_train_jax_fast_module()
    _check("train_jax_fast.py exposes _attempt_checkpoint_load",
          hasattr(tjf, "_attempt_checkpoint_load"))

    _check_empty_flag_gate(tjf)

    tmp = tempfile.mkdtemp(prefix="test888_train_jax_fast_")
    try:
        # --- the exact D38 item-2 scenario: a resume pointed at a bad checkpoint dir must
        # RAISE (killing the run with a nonzero exit via main()'s try/except: print; raise),
        # never silently continue on fresh-init params under a "loaded successfully" log ------
        trainer = _tiny_trainer(seed=0)
        params_before = trainer.state.params
        bad_path = os.path.join(tmp, "nonexistent_resume_checkpoint")
        try:
            tjf._attempt_checkpoint_load(trainer, bad_path, reset_optimizer=False)
            _check("nonexistent --load-checkpoint dir RAISES (never a silent fresh-init "
                  "resume)", False, "did not raise")
        except RuntimeError as e:
            _check("nonexistent --load-checkpoint dir RAISES, naming the resolved path",
                  os.path.realpath(bad_path) in str(e), str(e))
        _check("trainer.state.params UNCHANGED after the refused load",
              trainer.state.params is params_before, "params were mutated despite refusal")

        # --- a REAL, loadable checkpoint: must actually install it and return evidence ------
        ckpt_dir = os.path.join(tmp, "ckpt_real")
        trainer_src = _tiny_trainer(seed=1)
        trainer_src.save_checkpoint(ckpt_dir, overwrite=True)

        trainer2 = _tiny_trainer(seed=0)
        params_before2 = trainer2.state.params
        evidence = tjf._attempt_checkpoint_load(trainer2, ckpt_dir, reset_optimizer=False)
        _check("a real --load-checkpoint replaces trainer.state.params (object identity)",
              trainer2.state.params is not params_before2, "params object identity unchanged")
        _check("_attempt_checkpoint_load returns load evidence",
              evidence["loaded"] is True and isinstance(evidence["param_fingerprint"], str) and
              len(evidence["param_fingerprint"]) == 64, str(evidence))

        # --- --reset-optimizer threading: optimizer state resets, model weights untouched ---
        trainer3 = _tiny_trainer(seed=0)
        opt_state_before = trainer3.state.opt_state
        tjf._attempt_checkpoint_load(trainer3, ckpt_dir, reset_optimizer=True)
        _check("reset_optimizer=True actually calls reset_optimizer_state (opt_state object "
              "identity changes)", trainer3.state.opt_state is not opt_state_before,
              "opt_state unchanged despite reset_optimizer=True")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("=" * 84)
    if failures:
        print(f"[test_train_jax_fast_checkpoint_888] FAIL — {len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[test_train_jax_fast_checkpoint_888] PASS — resume load path refuses correctly and "
          "installs a real checkpoint (D38 item 2 fixed).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
