#!/usr/bin/env python3
"""TASK_888 — regression suite for the shared checkpoint resolver + fail-closed loader
(`mcts_trader/jax_agent/checkpoint_io.py`).

Covers acceptance gates (i) path identity, (ii) refusal-naming (generic; per-site refusal is
also exercised directly in each of the four call sites — see the task report), (iii) the
mutation-rebuilt negative control, and (iv) load-evidence/fingerprint stability. CPU-only,
no GPU, no `trading_data.db`, no fixture DB — every checkpoint here is a throwaway tiny model
saved to a tempdir and removed at the end.

Run: ``python mcts_trader/jax_agent/test_checkpoint_io.py`` (exit 0 = PASS).
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import shutil
import stat as stat_mod
import sys
import tempfile

import jax
import jax.numpy as jnp

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


def _tiny_trainer(n_assets=2, seed=0):
    """The smallest trainer this program's own construction path (TrainingConfig ->
    setup_model_and_trainer -> init_train_state) will build — plumbing tests only, no
    training/search. Mirrors eval_vector.py's build_model_and_trainer minus the CLI args
    indirection."""
    cfg = TrainingConfig(name="test_checkpoint_io", model_size="small", vector_action=True,
                         seed=seed)
    model, trainer = setup_model_and_trainer(cfg, n_assets)
    dummy_obs = jnp.ones((1, 27, n_assets, n_assets))
    trainer.init_train_state(jax.random.PRNGKey(seed), dummy_obs)
    return trainer


class _RecordingTrainer:
    """A minimal double for gate (i): records the EXACT path handed to `load_checkpoint`
    without ever opening anything (flax's own no-op-alias behavior — returning `target`
    unchanged — is simulated directly, so this never touches a real filesystem checkpoint).

    Mirrors the real `MuZeroTrainer.load_checkpoint(path, step=None, leaf_dir=None)` signature:
    `leaf_dir` is the resolved orbax leaf the loader must actually restore (fix-pass B1)."""
    def __init__(self):
        self.calls = []
        self.leaf_calls = []
        self.state = type("S", (), {"params": {"w": jnp.zeros(1)}, "step_count": 0})()

    def load_checkpoint(self, path, step=None, leaf_dir=None):
        self.calls.append(path)
        self.leaf_calls.append(leaf_dir)
        return self.state    # alias-back == "nothing loadable here" (measured flax behavior)


def test_gate_i_path_identity_synthetic():
    """gate (i), primary: a self-contained reproduction of the exact D38/c3 shape — a
    `--checkpoint` string crossing a symlink and then `..` — with a REAL checkpoint sitting at
    the REALPATH-resolved location and NOTHING at the ABSPATH-lexical location. Proves the
    guard (`resolve_checkpoint_dirs`) and the loader (`load_checkpoint_into`, which feeds
    `trainer.load_checkpoint`) agree on realpath end to end: the load actually SUCCEEDS through
    the crossing, which is only possible if both use the same resolution."""
    tmp = tempfile.mkdtemp(prefix="test888_gate_i_")
    try:
        outer = os.path.join(tmp, "outer")            # abspath would (wrongly) land here
        real = os.path.join(tmp, "real")               # realpath correctly lands here
        inner = os.path.join(real, "inner")
        os.makedirs(outer)
        os.makedirs(inner)
        link = os.path.join(outer, "jax_ckpt_fast")     # the shipped symlink, reproduced
        os.symlink(inner, link)
        crossing_path = os.path.join(link, "..")        # "crosses a symlink then .."

        abspath = os.path.abspath(crossing_path)
        realpath = os.path.realpath(crossing_path)
        _check("gate (i) setup: abspath and realpath of the crossing path actually diverge "
              "(otherwise this test would vacuously pass under the pre-888 bug too)",
              abspath != realpath, f"abspath={abspath} realpath={realpath}")
        _check("gate (i) setup: realpath lands on the directory that actually holds a "
              "checkpoint", realpath == real, f"realpath={realpath} expected={real}")
        _check("gate (i) setup: abspath lands on a DIFFERENT, checkpoint-less directory "
              "(the historical bug's directory B)", abspath == outer and abspath != real,
              f"abspath={abspath} outer={outer}")

        # Save a REAL checkpoint at `real` (the realpath-correct location) — nothing at `outer`.
        trainer_a = _tiny_trainer(seed=0)
        trainer_a.save_checkpoint(real, overwrite=True)

        run_dir, leaf_dir = cio.resolve_checkpoint_dirs(crossing_path)
        _check("gate (i): resolve_checkpoint_dirs (the guard) resolves the crossing path via "
              "realpath, finds the checkpoint at `real`", run_dir == realpath and
              leaf_dir is not None, f"run_dir={run_dir} leaf_dir={leaf_dir}")

        # THE loader: must feed trainer.load_checkpoint the SAME resolved directory (never a
        # separately-abspath'd one) — proven by an actual successful load through the crossing.
        trainer_b = _tiny_trainer(seed=1)
        params_before = trainer_b.state.params
        evidence = cio.load_checkpoint_into(trainer_b, crossing_path)
        _check("gate (i): load_checkpoint_into LOADS SUCCESSFULLY through the symlink+.. "
              "crossing (only possible if the loader used realpath, matching the guard)",
              evidence["loaded"] is True, f"evidence={evidence}")
        _check("gate (i): the loaded params are NOT the fresh-init params (a real load "
              "happened, not a silent no-op)",
              trainer_b.state.params is not params_before, "params object identity unchanged")
        _check("gate (i): load evidence's resolved_path sits under the REALPATH location, "
              "never under the abspath-lexical one",
              evidence["resolved_path"].startswith(real) and
              not evidence["resolved_path"].startswith(outer),
              f"resolved_path={evidence['resolved_path']}")

        # Negative control: a loader that (like the pre-888 code) fed ABSPATH instead of the
        # guard's resolution would have looked in `outer`, found nothing, and (pre-D37-fix)
        # silently kept the fresh init while reporting success. Confirm `outer` really is empty
        # of any loadable checkpoint, so this is a faithful reproduction of the historical bug's
        # consequence, not a vacuous setup.
        run_dir_outer, leaf_dir_outer = cio.resolve_checkpoint_dirs(outer)
        _check("gate (i) negative control: the abspath-lexical directory genuinely has NO "
              "loadable checkpoint (this is what the pre-888 loader would have silently "
              "no-op'ed against)", leaf_dir_outer is None, f"leaf_dir_outer={leaf_dir_outer}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_gate_i_shipped_symlink_layout():
    """gate (i), corroborating: the LITERAL shipped layout (`jax_ckpt_fast` ->
    `/mnt/scratch/...`). Read-only (stat/readlink only); skips (does not fail) if
    this tree/host does not have the symlink or its target mounted, since that is an
    environment fact, not a code defect."""
    link = os.path.join(_REPO, "jax_ckpt_fast")
    if not os.path.islink(link) or not os.path.isdir(link):
        print("  [SKIP] gate (i) shipped-layout check: jax_ckpt_fast is not a mounted symlink "
              "in this environment")
        return
    crossing_path = os.path.join(link, "..")
    abspath = os.path.abspath(crossing_path)
    realpath = os.path.realpath(crossing_path)
    if abspath == realpath:
        print("  [SKIP] gate (i) shipped-layout check: abspath==realpath here (unexpected "
              "topology) — the synthetic reproduction above is the load-bearing proof")
        return
    run_dir, _leaf = cio.resolve_checkpoint_dirs(crossing_path)
    _check("gate (i) shipped layout: resolve_checkpoint_dirs resolves jax_ckpt_fast/.. via "
          "realpath (the guard), not abspath",
          run_dir == realpath, f"run_dir={run_dir} realpath={realpath} abspath={abspath}")
    rec = _RecordingTrainer()
    try:
        cio.load_checkpoint_into(rec, crossing_path)
    except cio.CheckpointRefusalError:
        pass
    if rec.calls:
        _check("gate (i) shipped layout: the loader's actual trainer.load_checkpoint call used "
              "the SAME resolved directory as the guard",
              os.path.realpath(rec.calls[0]) == realpath,
              f"loader called with {rec.calls[0]!r}, guard resolved {realpath!r}")
        _check("gate (i) shipped layout: the loader was handed the RESOLVED LEAF, not left to "
              "re-pick one (B1)",
              rec.leaf_calls[0] == cio.resolve_checkpoint_dirs(crossing_path)[1],
              f"leaf handed to the loader: {rec.leaf_calls[0]!r}")
    else:
        print("  [INFO] gate (i) shipped layout: no leaf found at the resolved crossing path "
              "(refused before any trainer.load_checkpoint call) — consistent, not a failure")


def test_gate_ii_iv_refuse_and_fingerprint():
    """gate (ii)/(iv): refuses (naming the resolved path) on nonexistent/empty/mismatched
    checkpoints; on a real load, records resolved_path/step/param_fingerprint, and the
    fingerprint is stable across repeated loads of the SAME checkpoint and DIFFERENT across
    differently-trained ones."""
    tmp = tempfile.mkdtemp(prefix="test888_gate_ii_iv_")
    try:
        # --- (ii) nonexistent path: hard refusal naming the resolved path -----------------
        nonexistent = os.path.join(tmp, "does_not_exist")
        trainer = _tiny_trainer(seed=0)
        try:
            cio.load_checkpoint_into(trainer, nonexistent)
            _check("gate (ii): nonexistent --checkpoint REFUSES", False,
                  "load_checkpoint_into returned instead of raising")
        except cio.CheckpointRefusalError as e:
            _check("gate (ii): nonexistent --checkpoint REFUSES, naming the resolved path",
                  os.path.realpath(nonexistent) in str(e), f"message did not name the path: {e}")

        # --- (ii) empty directory (exists, nothing loadable): hard refusal -----------------
        empty_dir = os.path.join(tmp, "empty_ckpt_dir")
        os.makedirs(empty_dir)
        try:
            cio.load_checkpoint_into(trainer, empty_dir)
            _check("gate (ii): empty checkpoint dir REFUSES", False, "did not raise")
        except cio.CheckpointRefusalError:
            _check("gate (ii): empty checkpoint dir REFUSES", True)

        # --- allow_fresh_init: the ONLY way to bypass, and it changes nothing silently ------
        state_before = trainer.state
        evidence_fresh = cio.load_checkpoint_into(trainer, nonexistent, allow_fresh_init=True)
        _check("allow_fresh_init=True on a nonexistent path returns evidence without raising",
              evidence_fresh["loaded"] is False, f"evidence={evidence_fresh}")
        _check("allow_fresh_init=True never mutates trainer.state (fresh-init stays exactly "
              "what the caller built)", trainer.state is state_before, "state was replaced")

        # --- real save/load round trip: evidence fields + fingerprint stability ------------
        ckpt_a = os.path.join(tmp, "ckpt_a")
        trainer_a1 = _tiny_trainer(seed=0)
        trainer_a1.save_checkpoint(ckpt_a, overwrite=True)

        trainer_a2 = _tiny_trainer(seed=1)          # different init; must be overwritten by load
        ev_a2 = cio.load_checkpoint_into(trainer_a2, ckpt_a)
        _check("gate (iv): load evidence carries resolved_path", bool(ev_a2["resolved_path"]))
        _check("gate (iv): load evidence carries step", ev_a2["step"] == 0, str(ev_a2))
        _check("gate (iv): load evidence carries a param_fingerprint",
              isinstance(ev_a2["param_fingerprint"], str) and len(ev_a2["param_fingerprint"]) == 64,
              str(ev_a2))

        trainer_a3 = _tiny_trainer(seed=2)          # yet another fresh init, load the SAME ckpt
        ev_a3 = cio.load_checkpoint_into(trainer_a3, ckpt_a)
        _check("gate (iv): the SAME checkpoint loaded twice (different fresh-init trainers) "
              "fingerprints IDENTICALLY",
              ev_a2["param_fingerprint"] == ev_a3["param_fingerprint"],
              f"{ev_a2['param_fingerprint']} != {ev_a3['param_fingerprint']}")

        # A DIFFERENT checkpoint (trained/initialized differently) must fingerprint differently.
        ckpt_b = os.path.join(tmp, "ckpt_b")
        trainer_b1 = _tiny_trainer(seed=99)
        trainer_b1.save_checkpoint(ckpt_b, overwrite=True)
        trainer_b2 = _tiny_trainer(seed=0)
        ev_b2 = cio.load_checkpoint_into(trainer_b2, ckpt_b)
        _check("gate (iv): a DIFFERENT checkpoint fingerprints DIFFERENTLY",
              ev_a2["param_fingerprint"] != ev_b2["param_fingerprint"],
              "two differently-seeded checkpoints produced the same fingerprint")

        # --- foreign-shape restore (n_assets mismatch) REFUSES, never loads on anyway -------
        # Scoped exactly as `loaded_param_shape_defect` now claims (fix-pass B4): this is a
        # SHAPE check. Leaves the checkpoint has and the model does not (or vice versa) are
        # invisible to it by construction — flax returns a pytree congruent to the TARGET — and
        # the branches that pretended otherwise were unreachable and are gone.
        ckpt_2a = os.path.join(tmp, "ckpt_2assets")
        trainer_2a = _tiny_trainer(n_assets=2, seed=0)
        trainer_2a.save_checkpoint(ckpt_2a, overwrite=True)
        trainer_3a = _tiny_trainer(n_assets=3, seed=0)
        try:
            cio.load_checkpoint_into(trainer_3a, ckpt_2a)
            _check("gate (ii): an n_assets-mismatched checkpoint REFUSES (foreign shapes)",
                  False, "did not raise on a shape mismatch")
        except cio.CheckpointRefusalError as e:
            _check("gate (ii): an n_assets-mismatched checkpoint REFUSES (foreign shapes)",
                  "FOREIGN SHAPES" in str(e), str(e))

        # B4, stated as a TEST rather than only as prose: the leaf SETS are always equal after a
        # restore-into-target, so a "leaf present in one and not the other" check could never
        # fire. This is the fact that made the removed branches unreachable.
        keys = lambda t: {jax.tree_util.keystr(p)
                          for p, _v in jax.tree_util.tree_flatten_with_path(t)[0]}
        t_before = _tiny_trainer(n_assets=2, seed=3)
        before_keys = keys(t_before.state.params)
        cio.load_checkpoint_into(t_before, ckpt_2a)
        _check("B4: a restore-into-target returns a pytree with EXACTLY the target's leaf set "
              "(so `only_model`/`only_ckpt` were unreachable, not merely unexercised)",
              keys(t_before.state.params) == before_keys,
              "the restored leaf set differed from the target's")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_gate_iii_mutation_rebuilt_negative_control():
    """gate (iii): rebuild the REMOVED check independently (the pre-888/D37 shape — refuse only
    when flax returns `None`) and confirm it does NOT catch the no-op alias-back that
    `load_checkpoint_into`'s object-identity check DOES catch. Proves this suite is not
    vacuously green regardless of the fix: the old check and the new check must disagree on the
    exact scenario the fix targets."""
    class _AliasBackTrainer:
        """Simulates flax's measured no-op behavior: `restore_checkpoint` returns `target`
        (here: the trainer's own current state) UNCHANGED when nothing is loadable — never
        `None`, never a raise."""
        def __init__(self):
            self.state = type("S", (), {"params": {"w": jnp.zeros(1)}, "step_count": 0})()

        def load_checkpoint(self, path, step=None, leaf_dir=None):
            return self.state     # the measured flax alias-back, not None

    def _pre_888_check(trainer, path):
        """The MUTATION: the exact shape of the removed check (D37) — `if loaded is None:
        raise`, nothing else. This is what train_jax_fast.py / distill.py / real_trade_jax.py /
        eval_vector.py all did before this task."""
        loaded = trainer.load_checkpoint(path)
        if loaded is None:
            raise RuntimeError("no checkpoint")
        trainer.state = loaded
        return "loaded successfully"       # the historical lie

    t1 = _AliasBackTrainer()
    old_result = _pre_888_check(t1, "/does/not/matter")
    _check("gate (iii) negative control: the REMOVED pre-888 check ('is None' only) does NOT "
          "refuse on the alias-back no-op — this is the bug this task fixes",
          old_result == "loaded successfully", f"old_result={old_result!r}")

    t2 = _AliasBackTrainer()
    raised = False
    try:
        cio.load_checkpoint_into(t2, "/does/not/matter")
    except cio.CheckpointRefusalError:
        raised = True
    _check("gate (iii): the NEW check (object identity) DOES refuse on the identical "
          "alias-back scenario the old check missed — the mutation that removes this "
          "refusal (reverting to 'is None' only) goes red under this exact test",
          raised, "load_checkpoint_into did not raise on an alias-back no-op")


def test_checkpoint_leaf_defect_fifo_no_hang():
    """The hang guard: a FIFO planted inside a checkpoint leaf must be NAMED, never opened.
    `os.mkfifo` + `checkpoint_leaf_defect` only ever stat/scandir — if this test completes at
    all (no external timeout kill needed), the guard did not try to open the FIFO."""
    tmp = tempfile.mkdtemp(prefix="test888_fifo_")
    try:
        trainer = _tiny_trainer(seed=0)
        ckpt_dir = os.path.join(tmp, "ckpt_fifo")
        written = trainer.save_checkpoint(ckpt_dir, overwrite=True)
        # Find a real payload file inside the leaf and swap it for a FIFO.
        victim = None
        for root, _dirs, files in os.walk(written):
            for f in files:
                victim = os.path.join(root, f)
                break
            if victim:
                break
        _check("FIFO-hang setup: found a real payload file inside the checkpoint leaf to swap",
              victim is not None, f"no files found under {written}")
        if victim is not None:
            os.remove(victim)
            os.mkfifo(victim)
            defect = cio.checkpoint_leaf_defect(written)
            _check("checkpoint_leaf_defect NAMES a FIFO planted in the payload instead of "
                  "hanging (this line executing at all is part of the proof)",
                  defect is not None and "FIFO" in defect, f"defect={defect}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_b1_multi_leaf_run_dir():
    """fix-pass B1 — THE defect the whole module exists to prevent, one layer down: a run dir
    holding MORE THAN ONE `checkpoint_<step>` (the DEFAULT — `save_checkpoint(keep=3)`).

    Naming an older leaf explicitly must LOAD that leaf. Before the fix the loader was handed
    the RUN DIR and flax re-picked `latest_checkpoint(run_dir)`, so the guard blessed leaf A
    while the loader installed leaf B and the evidence reported `resolved_path`=A next to
    `step`/`param_fingerprint`=B. This asserts the three things that falsifies:
      * the evidence's `step` is the step of the leaf that was ASKED for,
      * the two leaves fingerprint DIFFERENTLY (gate (iv): different checkpoint => different
        fingerprint — the claim that was false),
      * asking for the run dir itself still gets the LATEST leaf (unchanged semantics)."""
    tmp = tempfile.mkdtemp(prefix="test888_multileaf_")
    try:
        run_dir = os.path.join(tmp, "ckpt_multi")
        # Two genuinely different checkpoints (different inits) at two different steps, in ONE
        # run dir — exactly what a training run leaves behind under keep=3.
        t100 = _tiny_trainer(seed=0)
        t100.state = t100.state.replace(step_count=100)
        t100.save_checkpoint(run_dir, overwrite=True)
        t200 = _tiny_trainer(seed=7)
        t200.state = t200.state.replace(step_count=200)
        t200.save_checkpoint(run_dir, overwrite=True)
        leaves = sorted(d for d in os.listdir(run_dir) if d.startswith("checkpoint_"))
        _check("B1 setup: the run dir really holds TWO checkpoint_<step> leaves",
              leaves == ["checkpoint_100", "checkpoint_200"], f"leaves={leaves}")

        ev = {}
        for step in (100, 200):
            leaf = os.path.join(run_dir, f"checkpoint_{step}")
            trainer = _tiny_trainer(seed=1)
            ev[step] = cio.load_checkpoint_into(trainer, leaf)
            _check(f"B1: asking for checkpoint_{step} by name LOADS it (evidence step matches)",
                  ev[step]["step"] == step, f"evidence={ev[step]}")
            _check(f"B1: evidence resolved_path for checkpoint_{step} is that leaf",
                  ev[step]["resolved_path"] == os.path.realpath(leaf),
                  f"resolved_path={ev[step]['resolved_path']}")
        _check("B1/gate (iv): two DIFFERENT leaves in the SAME run dir fingerprint DIFFERENTLY "
              "(the pre-fix loader returned the identical fingerprint for both)",
              ev[100]["param_fingerprint"] != ev[200]["param_fingerprint"],
              f"both fingerprinted {ev[100]['param_fingerprint']}")

        trainer_run = _tiny_trainer(seed=1)
        ev_run = cio.load_checkpoint_into(trainer_run, run_dir)
        _check("B1: naming the RUN DIR still loads the LATEST leaf (semantics unchanged)",
              ev_run["step"] == 200 and
              ev_run["param_fingerprint"] == ev[200]["param_fingerprint"], f"evidence={ev_run}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_b2_run_dir_sidecar_fifo():
    """fix-pass B2 — the sidecars `trainer.load_checkpoint` opens RAW (`trainer_metadata.json`,
    `popart_state.npz`) are SIBLINGS of the orbax leaf, one level up, so the leaf walk never saw
    them: a FIFO at either passed every guard with no defect and then blocked `openat` FOREVER.
    Not a race — 100% reproducible. This test completing at all is part of the proof."""
    for name in cio.RUN_DIR_SIDECARS:
        tmp = tempfile.mkdtemp(prefix="test888_sidecar_")
        try:
            trainer = _tiny_trainer(seed=0)
            run_dir = os.path.join(tmp, "ckpt_sidecar")
            trainer.save_checkpoint(run_dir, overwrite=True)
            sidecar = os.path.join(run_dir, name)
            if os.path.exists(sidecar):
                os.remove(sidecar)
            os.mkfifo(sidecar)
            _check(f"B2: sidecar_defect NAMES a FIFO at {name} (leaf walk cannot see it)",
                  (cio.sidecar_defect(run_dir) or "").find(name) >= 0,
                  f"defect={cio.sidecar_defect(run_dir)!r}")
            t2 = _tiny_trainer(seed=1)
            try:
                cio.load_checkpoint_into(t2, run_dir)
                _check(f"B2: load_checkpoint_into REFUSES a FIFO at {name}", False,
                      "did not raise")
            except cio.CheckpointRefusalError as e:
                _check(f"B2: load_checkpoint_into REFUSES a FIFO at {name}, naming it",
                      name in str(e), str(e))
            try:
                cio.preflight_checkpoint_loadable([run_dir])
                _check(f"B2: preflight REFUSES a FIFO at {name}", False, "did not raise")
            except RuntimeError as e:
                _check(f"B2: preflight REFUSES a FIFO at {name}, naming it", name in str(e),
                      str(e))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def test_m2_symlinked_file_escape():
    """fix-pass M2: containment ran only on the S_ISDIR branch, so a symlink to a REGULAR FILE
    anywhere on the host was blessed unconditionally while a directory-typed escape was
    correctly refused. Both must be refused: an out-of-tree file is a path the loader will open
    that this guard cannot bound."""
    tmp = tempfile.mkdtemp(prefix="test888_escape_")
    try:
        outside = os.path.join(tmp, "outside")
        os.makedirs(outside)
        outside_file = os.path.join(outside, "not_mine.bin")
        with open(outside_file, "wb") as fh:
            fh.write(b"x")
        trainer = _tiny_trainer(seed=0)
        run_dir = os.path.join(tmp, "ckpt_escape")
        leaf = trainer.save_checkpoint(run_dir, overwrite=True)
        _check("M2 control: the untouched leaf walks clean",
              cio.unsafe_tree_defect(leaf) is None, f"{cio.unsafe_tree_defect(leaf)}")
        os.symlink(outside_file, os.path.join(leaf, "escaped_file"))
        defect = cio.unsafe_tree_defect(leaf)
        _check("M2: a symlink to a regular file OUTSIDE the leaf is REFUSED (it used to be "
              "blessed unconditionally on the S_ISREG branch)",
              defect is not None and "OUTSIDE" in defect, f"defect={defect!r}")
        os.remove(os.path.join(leaf, "escaped_file"))
        os.symlink(outside, os.path.join(leaf, "escaped_dir"))
        defect_dir = cio.unsafe_tree_defect(leaf)
        _check("M2 control: the directory-typed escape stays refused",
              defect_dir is not None and "OUTSIDE" in defect_dir, f"defect={defect_dir!r}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_m4_refusal_echo_is_clipped():
    """fix-pass M4: a ~130,000-char `--checkpoint` (legal — under MAX_ARG_STRLEN) produced a
    260,505-char exception and a 390,559-char log line. The refusal must still fire and still
    identify the path; only the ECHO is bounded."""
    huge = "/tmp/" + ("a" * 130_000)
    trainer = _tiny_trainer(seed=0)
    try:
        cio.load_checkpoint_into(trainer, huge)
        _check("M4: a huge --checkpoint still REFUSES", False, "did not raise")
    except cio.CheckpointRefusalError as e:
        msg = str(e)
        _check("M4: a huge --checkpoint still REFUSES", True)
        _check("M4: the refusal message is bounded (was 260,505 chars for this input)",
              len(msg) < 4000, f"message was {len(msg)} chars")
        _check("M4: the clip is DISCLOSED and counted, never silent",
              "[clipped," in msg, msg[:400])
        _check("M4: the refusal still identifies the path it refused",
              "/tmp/aaaa" in msg, msg[:400])
    try:
        cio.preflight_checkpoint_loadable([huge])
        _check("M4: preflight on a huge --checkpoint still REFUSES", False, "did not raise")
    except RuntimeError as e:
        _check("M4: preflight's refusal is bounded too", len(str(e)) < 4000,
              f"message was {len(str(e))} chars")


def test_minor_auto_discovered_leaf_escape():
    """fix-pass MINOR: `flax_checkpoints.latest_checkpoint` joins a bare `os.listdir` basename
    onto the run dir without realpathing the FINAL component, so a `checkpoint_0` SYMLINK inside
    run dir A pointing at a well-formed checkpoint elsewhere is auto-discovered and loaded — the
    payload comes from B while `trainer_metadata.json`/`popart_state.npz` are read from A.
    Nothing hangs and the fingerprint stays accurate (so it is not a fail-open), but guard and
    loader are then talking about two different checkpoints.

    The control matters as much as the refusal: when argv NAMES the leaf directly, `run_dir` is
    that leaf's own realpath'd parent, so this check must stay silent — an operator who points
    --checkpoint at a leaf has named the bytes they want."""
    tmp = tempfile.mkdtemp(prefix="test888_leafescape_")
    try:
        trainer = _tiny_trainer(seed=0)
        real_run = os.path.join(tmp, "elsewhere", "ckpt_real")
        real_leaf = trainer.save_checkpoint(real_run, overwrite=True)
        decoy = os.path.join(tmp, "ckpt_decoy")
        os.makedirs(decoy)
        os.symlink(real_leaf, os.path.join(decoy, "checkpoint_0"))

        run_dir, leaf_dir = cio.resolve_checkpoint_dirs(decoy)
        _check("MINOR: the planted symlink IS auto-discovered as the run dir's leaf (the "
               "precondition — flax joins the bare listdir name)",
               leaf_dir == os.path.join(run_dir, "checkpoint_0"),
               f"run_dir={run_dir} leaf={leaf_dir}")
        _check("MINOR: every OTHER guard passes it (this is not a hang and not a fail-open — "
               "the tree walk realpaths its own root, so the foreign payload walks clean)",
               cio.checkpoint_leaf_defect(leaf_dir) is None,
               f"{cio.checkpoint_leaf_defect(leaf_dir)}")
        escape = cio.leaf_escape_defect(run_dir, leaf_dir)
        _check("MINOR: leaf_escape_defect REFUSES an auto-discovered leaf that resolves outside "
               "its run dir", escape is not None and "OUTSIDE" in escape, f"defect={escape!r}")

        params_before = trainer.state.params
        try:
            cio.load_checkpoint_into(trainer, decoy)
            _check("MINOR: the loader refuses it", False, "loaded the external payload")
        except cio.CheckpointRefusalError as e:
            _check("MINOR: the loader refuses it, naming the escape", "OUTSIDE" in str(e),
                   str(e)[:400])
        _check("MINOR: trainer.state untouched by the refusal",
               trainer.state.params is params_before, "params were mutated despite refusal")
        try:
            cio.preflight_checkpoint_loadable([decoy])
            _check("MINOR: the preflight refuses it too", False, "did not raise")
        except RuntimeError as e:
            _check("MINOR: the preflight refuses it too", "OUTSIDE" in str(e), str(e)[:400])

        # CONTROL: argv naming the leaf directly must still load — containment is trivially
        # true there (run_dir = dirname(realpath(leaf))), so this check cannot fire.
        trainer2 = _tiny_trainer(seed=1)
        ev = cio.load_checkpoint_into(trainer2, os.path.join(decoy, "checkpoint_0"))
        _check("MINOR control: --checkpoint naming that same leaf DIRECTLY still loads (the "
               "check only bites on auto-discovery)", ev["loaded"] is True, str(ev))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_preflight_checkpoint_loadable():
    tmp = tempfile.mkdtemp(prefix="test888_preflight_")
    try:
        trainer = _tiny_trainer(seed=0)
        ckpt_dir = os.path.join(tmp, "ckpt_ok")
        trainer.save_checkpoint(ckpt_dir, overwrite=True)
        try:
            cio.preflight_checkpoint_loadable([ckpt_dir])
            _check("preflight_checkpoint_loadable passes a real checkpoint", True)
        except cio.CheckpointRefusalError as e:
            _check("preflight_checkpoint_loadable passes a real checkpoint", False, str(e))

        bad = os.path.join(tmp, "nope")
        try:
            cio.preflight_checkpoint_loadable([ckpt_dir, bad])
            _check("preflight_checkpoint_loadable refuses when ANY entry in the list is bad",
                  False, "did not raise")
        except RuntimeError as e:
            _check("preflight_checkpoint_loadable refuses when ANY entry in the list is bad, "
                  "naming it", bad in str(e) or os.path.realpath(bad) in str(e), str(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_fixpass_unreadable_payload_refuses_typed():
    """fix-pass (read-permission gap): the guards checked `S_ISREG`/`S_ISDIR`/size but never
    READABILITY, so a `chmod 000` file inside an otherwise genuine leaf passed everything and the
    failure then surfaced from orbax/tensorstore as a bare
    `ValueError: PERMISSION_DENIED: Error opening "zarr" driver ...` — NOT a
    `CheckpointRefusalError`. That un-uniformed the refusal for callers that narrowly catch this
    module's own type (`tools/paper_trade_vectorbt.py`'s `except CheckpointRefusalError`, whose
    comment claims "same refusal — only legible"). Type says the open will not BLOCK; it does not
    say it will SUCCEED."""
    if os.geteuid() == 0:
        print("  [SKIP] unreadable-payload checks (running as root: os.access is vacuously true)")
        return
    # a) an orbax MARKER inside the leaf, and b) a RUN-DIR SIDECAR the loader opens raw
    for victim_name, expect_in_msg in (("manifest.ocdbt", "manifest.ocdbt"),
                                       ("trainer_metadata.json", "trainer_metadata.json")):
        tmp = tempfile.mkdtemp(prefix="test888_unreadable_")
        victim = None
        try:
            trainer = _tiny_trainer(seed=0)
            run_dir = os.path.join(tmp, "ckpt_unreadable")
            leaf = trainer.save_checkpoint(run_dir, overwrite=True)
            victim = (os.path.join(leaf, victim_name) if victim_name == "manifest.ocdbt"
                      else os.path.join(run_dir, victim_name))
            if not os.path.exists(victim):
                print(f"  [SKIP] {victim_name} not present in this checkpoint")
                continue
            t_ok = _tiny_trainer(seed=1)
            _check(f"control: the checkpoint loads cleanly before {victim_name} is chmod'ed",
                  cio.load_checkpoint_into(t_ok, run_dir)["loaded"] is True, "control load failed")
            os.chmod(victim, 0o000)
            t2 = _tiny_trainer(seed=1)
            try:
                cio.load_checkpoint_into(t2, run_dir)
                _check(f"unreadable {victim_name} is REFUSED", False,
                      "loaded despite an unreadable payload file")
            except cio.CheckpointRefusalError as e:
                _check(f"unreadable {victim_name} raises CheckpointRefusalError (not a raw "
                      f"ValueError from tensorstore), naming the file",
                      expect_in_msg in str(e) and "READABLE" in str(e), str(e)[:400])
            except Exception as e:                     # the pre-fix behaviour, now a failure
                _check(f"unreadable {victim_name} raises CheckpointRefusalError, not "
                      f"{type(e).__name__}", False, f"{type(e).__name__}: {str(e)[:200]}")
            try:
                cio.preflight_checkpoint_loadable([run_dir])
                _check(f"preflight REFUSES an unreadable {victim_name}", False, "did not raise")
            except RuntimeError as e:
                _check(f"preflight REFUSES an unreadable {victim_name}, naming it",
                      expect_in_msg in str(e), str(e)[:400])
        finally:
            if victim and os.path.exists(victim):
                os.chmod(victim, 0o644)                # else shutil.rmtree cannot clean up
            shutil.rmtree(tmp, ignore_errors=True)


def test_fixpass_clip_discipline_is_uniform():
    """fix-pass (clip discipline): `unsafe_to_open_defect`, `unsafe_tree_defect` and
    `checkpoint_leaf_defect` interpolated raw `OSError` text (`{e}`, which embeds the offending
    path VERBATIM) and `{rel}` directly, bypassing `clip_for_refusal` — unlike
    `leaf_escape_defect` and the two top-level refusals, which clip every variable part. Bounded
    by PATH_MAX in practice, so this is a consistency defect rather than a live flood, but it
    contradicted the module's own stated intent. Every variable part of every defect phrase must
    now be clipped and the clip DISCLOSED."""
    tmp = tempfile.mkdtemp(prefix="test888_clip_")
    try:
        trainer = _tiny_trainer(seed=0)
        run_dir = os.path.join(tmp, "ckpt_clip")
        leaf = trainer.save_checkpoint(run_dir, overwrite=True)
        # A deep, long-named subdirectory: its `rel` is what the walk interpolates.
        long_rel = os.path.join(leaf, "z" * 200, "y" * 200)
        os.makedirs(long_rel)
        os.mkfifo(os.path.join(long_rel, "w" * 200))
        defect = cio.unsafe_tree_defect(leaf)
        _check("clip: a long entry path inside the leaf is still REFUSED",
              defect is not None, "no defect reported for a FIFO under a long path")
        _check("clip: the walk's defect phrase is bounded and the clip is disclosed",
              defect is not None and len(defect) < 1000 and "[clipped," in defect,
              f"len={len(defect or '')} phrase={str(defect)[:300]}")
        # `unsafe_to_open_defect`'s stat-failure branch: the OSError text embeds the offending
        # path VERBATIM, so a long path echoes at full length unless clipped. Components stay
        # under NAME_MAX (255) and the total under PATH_MAX (4096) so the failure is a plain
        # ENOENT carrying the whole path, not an ENAMETOOLONG carrying almost none of it.
        long_missing = os.path.join(tmp, *(["q" * 200] * 12), "missing_sidecar")
        _check("clip setup: the probe path really is long (else nothing needs clipping)",
              len(long_missing) > 2000, f"len={len(long_missing)}")
        phrase = cio.unsafe_to_open_defect(long_missing, 4096, "a test sidecar")
        _check("clip: unsafe_to_open_defect's OSError echo is bounded and disclosed",
              phrase is not None and len(phrase) < 500 and "[clipped," in phrase,
              f"len={len(phrase or '')} phrase={str(phrase)[:300]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    print("[test_checkpoint_io] gate (i): path identity (symlink + '..')")
    test_gate_i_path_identity_synthetic()
    test_gate_i_shipped_symlink_layout()
    print("[test_checkpoint_io] gate (ii)/(iv): refusal + load evidence/fingerprint")
    test_gate_ii_iv_refuse_and_fingerprint()
    print("[test_checkpoint_io] gate (iii): mutation-rebuilt negative control")
    test_gate_iii_mutation_rebuilt_negative_control()
    print("[test_checkpoint_io] B1: multi-leaf run dir — the evidence and the load must agree")
    test_b1_multi_leaf_run_dir()
    print("[test_checkpoint_io] hang guard: FIFO in checkpoint payload")
    test_checkpoint_leaf_defect_fifo_no_hang()
    print("[test_checkpoint_io] B2: FIFO at a RUN-DIR sidecar (outside the leaf walk)")
    test_b2_run_dir_sidecar_fifo()
    print("[test_checkpoint_io] M2: symlink escape, files as well as directories")
    test_m2_symlinked_file_escape()
    print("[test_checkpoint_io] M4: bounded refusal echo")
    test_m4_refusal_echo_is_clipped()
    print("[test_checkpoint_io] MINOR: auto-discovered leaf must live in its run dir")
    test_minor_auto_discovered_leaf_escape()
    print("[test_checkpoint_io] preflight_checkpoint_loadable")
    test_preflight_checkpoint_loadable()
    print("[test_checkpoint_io] fix-pass: unreadable payload refuses with the NAMED type")
    test_fixpass_unreadable_payload_refuses_typed()
    print("[test_checkpoint_io] fix-pass: clip discipline uniform across defect phrases")
    test_fixpass_clip_discipline_is_uniform()

    print("=" * 84)
    if failures:
        print(f"[test_checkpoint_io] FAIL — {len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[test_checkpoint_io] PASS — all gates green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
