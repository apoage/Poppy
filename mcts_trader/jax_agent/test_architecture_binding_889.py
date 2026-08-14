#!/usr/bin/env python3
"""TASK_889 (D39) — regression suite for checkpoint ARCHITECTURE BINDING.

Companion to `test_checkpoint_io.py` (TASK_888's path-identity/object-identity/hang-guard
suite, unmodified by this task). This file pins the layer 888R deliberately did not scope:
`vector_action`/`use_prediction_head`/`param_fingerprint` recorded at SAVE time
(`trainer.save_checkpoint`), and `trainer.load_checkpoint` (called through
`checkpoint_io.load_checkpoint_into`) REFUSING when a checkpoint's recorded architecture or
fingerprint disagrees with the target it is loading into — never a print-and-continue.

Gates covered (task file numbering):
  (i)   save writes vector_action/use_prediction_head/param_fingerprint; a matching target
        loads successfully, a mismatched one REFUSES naming both values.
  (ii)  the specific halt case, CONSTRUCTED (not the shipped config): a prediction-head-enabled
        target vs a `use_prediction_head: false` checkpoint REFUSES.
  (iii) a checkpoint whose payload is altered after save produces a fingerprint disagreement
        and REFUSES; the disagreement is never overwritten (this method writes nothing at
        load time at all — the assertion is that the file on disk is untouched).
  (vi)  byte-identity: a correctly-matching load reproduces the saved params EXACTLY.

Every checkpoint here is a throwaway tiny (n_assets=2) model in a tempdir. CPU-only, no GPU, no
`trading_data.db`, no fixture DB, nothing under `runs/` touched.

Run: ``python mcts_trader/jax_agent/test_architecture_binding_889.py`` (exit 0 = PASS).
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import json
import shutil
import sys
import tempfile

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import jax
import jax.numpy as jnp
import numpy as np

from mcts_trader.training.config import TrainingConfig
from mcts_trader.training.setup import setup_model_and_trainer
from mcts_trader.jax_agent import checkpoint_io as cio

failures = []


def _check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        failures.append(f"{name}: {detail}")


def _tiny_trainer(n_assets=2, seed=0, vector_action=True, use_prediction_head=False):
    cfg = TrainingConfig(name="test_arch_binding_889", model_size="small",
                         vector_action=vector_action,
                         use_prediction_loss=use_prediction_head, seed=seed)
    _model, trainer = setup_model_and_trainer(cfg, n_assets)
    trainer.init_train_state(jax.random.PRNGKey(seed), jnp.ones((1, 27, n_assets, n_assets)))
    return trainer


def _read_meta(run_dir):
    with open(os.path.join(run_dir, "trainer_metadata.json")) as fh:
        return json.load(fh)


def _write_meta(run_dir, meta):
    with open(os.path.join(run_dir, "trainer_metadata.json"), "w") as fh:
        json.dump(meta, fh)


def _read_leaf_meta(run_dir, step):
    with open(cio.leaf_metadata_path_for(run_dir, step)) as fh:
        return json.load(fh)


def _strip_leaf_records(run_dir):
    """Delete every per-leaf `trainer_metadata_<step>.json` — what a run dir written BEFORE this
    task's save-time change looks like on disk. Used to construct the legacy cases; without it a
    "legacy" fixture built by the CURRENT save_checkpoint still carries the very record whose
    absence is the thing under test."""
    for name in os.listdir(run_dir):
        if name.startswith("trainer_metadata_") and name.endswith(".json"):
            os.remove(os.path.join(run_dir, name))


def _make_legacy(run_dir):
    """A checkpoint indistinguishable from a pre-TASK_889 save: no per-leaf record, and neither
    `vector_action` nor `param_fingerprint` in the shared sidecar."""
    _strip_leaf_records(run_dir)
    meta = _read_meta(run_dir)
    _write_meta(run_dir, {k: v for k, v in meta.items()
                          if k not in ("vector_action", "param_fingerprint")})


def _leaf_dict(params):
    return {jax.tree_util.keystr(p): np.asarray(v) for p, v in
            jax.tree_util.tree_flatten_with_path(params)[0]}


def test_gate_i_save_records_architecture_and_fingerprint():
    tmp = tempfile.mkdtemp(prefix="test889_arch_save_")
    try:
        trainer = _tiny_trainer(vector_action=True, use_prediction_head=False, seed=0)
        ckpt = os.path.join(tmp, "ckpt")
        trainer.save_checkpoint(ckpt, overwrite=True)
        meta = _read_meta(ckpt)
        _check("gate (i): save writes vector_action", meta.get("vector_action") is True,
              str(meta))
        _check("gate (i): save writes use_prediction_head",
              meta.get("use_prediction_head") is False, str(meta))
        _check("gate (i): save writes a 64-hex-char param_fingerprint",
              isinstance(meta.get("param_fingerprint"), str)
              and len(meta["param_fingerprint"]) == 64, str(meta))
        _check("gate (i): the recorded fingerprint matches checkpoint_io.param_fingerprint of "
              "the ACTUAL saved params",
              meta["param_fingerprint"] == cio.param_fingerprint(trainer.state.params),
              "recorded fingerprint does not describe the params actually written")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_gate_i_matching_target_loads_successfully():
    tmp = tempfile.mkdtemp(prefix="test889_arch_match_")
    try:
        src = _tiny_trainer(vector_action=True, use_prediction_head=False, seed=1)
        ckpt = os.path.join(tmp, "ckpt")
        src.save_checkpoint(ckpt, overwrite=True)

        dst = _tiny_trainer(vector_action=True, use_prediction_head=False, seed=0)
        params_before = dst.state.params
        evidence = cio.load_checkpoint_into(dst, ckpt)
        _check("gate (i): a matching target loads successfully",
              evidence["loaded"] is True, str(evidence))
        _check("gate (i): dst.state.params is REPLACED (object identity)",
              dst.state.params is not params_before, "params object identity unchanged")
        _check("gate (i): the evidence fingerprint equals the checkpoint's own",
              evidence["param_fingerprint"] == cio.param_fingerprint(src.state.params),
              str(evidence))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_gate_i_mismatched_vector_action_refuses_naming_both():
    tmp = tempfile.mkdtemp(prefix="test889_arch_mismatch_va_")
    try:
        src = _tiny_trainer(vector_action=True, use_prediction_head=False, seed=1)
        ckpt = os.path.join(tmp, "ckpt")
        src.save_checkpoint(ckpt, overwrite=True)

        dst = _tiny_trainer(vector_action=False, use_prediction_head=False, seed=0)
        params_before = dst.state.params
        try:
            cio.load_checkpoint_into(dst, ckpt)
            _check("gate (i): a mismatched vector_action target REFUSES", False,
                  "did not raise")
        except RuntimeError as e:
            msg = str(e)
            _check("gate (i): ... naming the RECORDED value (True)", "vector_action=True" in msg,
                  msg)
            _check("gate (i): ... naming the TARGET value (False)", "vector_action=False" in msg,
                  msg)
            _check("gate (i): ... naming the resolved path", os.path.realpath(ckpt) in msg
                  or ckpt in msg, msg)
        _check("gate (i): the refusal leaves dst.state untouched",
              dst.state.params is params_before, "params mutated despite refusal")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_gate_ii_prediction_head_enabled_target_vs_false_checkpoint_refuses():
    """gate (ii): the EXACT halt case, CONSTRUCTED here rather than relying on the shipped
    config (which scope 6 flips to False, removing the landmine from config/real_trade_config.py
    but not from the regression suite — this is that constructed case)."""
    tmp = tempfile.mkdtemp(prefix="test889_arch_gate2_")
    try:
        src = _tiny_trainer(vector_action=True, use_prediction_head=False, seed=1)
        ckpt = os.path.join(tmp, "ckpt")
        src.save_checkpoint(ckpt, overwrite=True)
        meta = _read_meta(ckpt)
        _check("gate (ii) precondition: the checkpoint records use_prediction_head=false",
              meta.get("use_prediction_head") is False, str(meta))

        dst = _tiny_trainer(vector_action=True, use_prediction_head=True, seed=0)
        params_before = dst.state.params
        try:
            cio.load_checkpoint_into(dst, ckpt)
            _check("gate (ii): a prediction-head-enabled target vs a "
                  "use_prediction_head:false checkpoint REFUSES", False, "did not raise")
        except RuntimeError as e:
            msg = str(e)
            _check("gate (ii): ... naming the RECORDED value (False)",
                  "use_prediction_head=False" in msg, msg)
            _check("gate (ii): ... naming the TARGET value (True)",
                  "use_prediction_head=True" in msg, msg)
        _check("gate (ii): the refusal leaves dst.state untouched",
              dst.state.params is params_before, "params mutated despite refusal")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_gate_iii_tampered_payload_produces_fingerprint_disagreement():
    """gate (iii): the checkpoint's PAYLOAD (not just its metadata) changes underneath its own
    recorded evidence. Simulated by physically swapping the orbax leaf of checkpoint A for a
    DIFFERENT trainer's leaf (checkpoint B, same architecture, different seed => different
    values), while leaving A's trainer_metadata.json (recording A's original fingerprint)
    untouched — exactly what "the payload changed after save" looks like on disk."""
    tmp = tempfile.mkdtemp(prefix="test889_arch_gate3_")
    try:
        trainer_a = _tiny_trainer(vector_action=True, use_prediction_head=False, seed=0)
        ckpt_a = os.path.join(tmp, "ckpt_a")
        trainer_a.save_checkpoint(ckpt_a, overwrite=True)
        fp_a = _read_meta(ckpt_a)["param_fingerprint"]

        trainer_b = _tiny_trainer(vector_action=True, use_prediction_head=False, seed=99)
        ckpt_b = os.path.join(tmp, "ckpt_b")
        trainer_b.save_checkpoint(ckpt_b, overwrite=True)
        fp_b = cio.param_fingerprint(trainer_b.state.params)
        _check("gate (iii) precondition: the two checkpoints have DIFFERENT fingerprints "
              "(different seeds => different weights)", fp_a != fp_b, f"{fp_a} == {fp_b}")

        # Swap A's orbax leaf for B's, leaving A's trainer_metadata.json (fp_a) in place.
        leaf_a = os.path.join(ckpt_a, "checkpoint_0")
        leaf_b = os.path.join(ckpt_b, "checkpoint_0")
        shutil.rmtree(leaf_a)
        shutil.copytree(leaf_b, leaf_a)

        dst = _tiny_trainer(vector_action=True, use_prediction_head=False, seed=0)
        params_before = dst.state.params
        meta_before_bytes = open(os.path.join(ckpt_a, "trainer_metadata.json"), "rb").read()
        try:
            cio.load_checkpoint_into(dst, ckpt_a)
            _check("gate (iii): a tampered payload produces a fingerprint disagreement and "
                  "REFUSES", False, "did not raise")
        except RuntimeError as e:
            msg = str(e)
            _check("gate (iii): ... naming the RECORDED fingerprint", fp_a in msg, msg)
            _check("gate (iii): ... naming the RECOMPUTED (tampered) fingerprint", fp_b in msg,
                  msg)
        _check("gate (iii): the refusal leaves dst.state untouched",
              dst.state.params is params_before, "params mutated despite refusal")
        _check("gate (iii): the disagreement is NEVER overwritten — trainer_metadata.json on "
              "disk is byte-identical to before the load attempt",
              open(os.path.join(ckpt_a, "trainer_metadata.json"), "rb").read()
              == meta_before_bytes, "metadata was rewritten by a load-time refusal")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_legacy_checkpoint_without_recorded_fields_loads_without_refusing():
    """A checkpoint that predates TASK_889 (neither `vector_action` nor `param_fingerprint`
    recorded, and no per-leaf record) must NOT refuse on that basis alone — `None` means "cannot
    verify", never "recorded a specific wrong value". Backward compatibility for the 7 real
    survivors, none of which carry `vector_action` today.

    But it is no longer SILENT (fix pass): nothing on disk backs the payload that was just
    restored, so the evidence dict says so and stdout carries a ⚠️ WARNING. The load proceeds
    only because there is no evidence to be stale — a leaf whose run dir DOES hold evidence
    about another save refuses instead (`test_fixpass_legacy_multi_leaf_*` below)."""
    tmp = tempfile.mkdtemp(prefix="test889_arch_legacy_")
    try:
        src = _tiny_trainer(vector_action=True, use_prediction_head=False, seed=1)
        ckpt = os.path.join(tmp, "ckpt")
        src.save_checkpoint(ckpt, overwrite=True)
        _make_legacy(ckpt)

        # Target's OWN vector_action can be anything — there is nothing recorded to disagree
        # with, so this must succeed regardless (a pre-889 checkpoint cannot be second-guessed
        # on a fact it never recorded).
        dst = _tiny_trainer(vector_action=True, use_prediction_head=False, seed=0)
        evidence = cio.load_checkpoint_into(dst, ckpt)
        _check("legacy checkpoint (no vector_action/param_fingerprint) loads without refusing",
              evidence["loaded"] is True, str(evidence))
        _check("fix pass: ... but the evidence dict MARKS it as unverified (loaded=True alone "
              "used to be indistinguishable from a verified load at every call site)",
              isinstance(evidence.get("verification_skipped"), str)
              and evidence.get("fingerprint_verified") is False, str(evidence))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_fixpass_verified_load_evidence_carries_no_extra_keys():
    """The verification keys appear ONLY in the degraded case, and that is load-bearing, not a
    style choice: this dict is embedded verbatim as `checkpoint_load` in eval_vector.py's
    artifact, and 861/smoke_e2e.py leg (i) compares that artifact BYTE-FOR-BYTE against a
    committed baseline with only `resolved_path` normalized. Unconditional new keys would turn a
    green leg red for a fully-verified load and demand a planner-gated baseline promotion that
    says nothing new. (Same idiom as TASK_883's `universe_pinned: false`.)"""
    tmp = tempfile.mkdtemp(prefix="test889_arch_evidence_")
    try:
        src = _tiny_trainer(vector_action=True, use_prediction_head=False, seed=1)
        ckpt = os.path.join(tmp, "ckpt")
        src.save_checkpoint(ckpt, overwrite=True)
        dst = _tiny_trainer(vector_action=True, use_prediction_head=False, seed=0)
        evidence = cio.load_checkpoint_into(dst, ckpt)
        _check("fix pass: a FULLY VERIFIED load returns exactly the four committed-baseline "
              "keys (no schema drift into the graded artifact)",
              set(evidence) == {"resolved_path", "step", "param_fingerprint", "loaded"},
              f"extra/missing keys: {set(evidence) ^ {'resolved_path', 'step', 'param_fingerprint', 'loaded'}}")
        _check("fix pass: ... and `verification_skipped` is absent, which is the machine-visible "
              "test for 'this load was verified'",
              evidence.get("verification_skipped") is None, str(evidence))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_fixpass_multi_leaf_heterogeneous_architecture_refuses():
    """THE BLOCKER, reproduced through the production API and closed.

    A run dir holding MORE THAN ONE `checkpoint_<step>` is the NORM, not the exception —
    `keep=3` is `save_checkpoint`'s default and what `train_jax_fast.py` passes, and
    `tools/paper_trade_vectorbt.py`'s own --help example names a specific leaf. The shared
    `trainer_metadata.json` is overwritten by the LATEST save, so for an older leaf it describes
    a DIFFERENT architecture. The first draft answered a `step_count` disagreement by SKIPPING
    all three checks — so loading a `vector_action=True` leaf into a `vector_action=False`
    target gave NO refusal, `vec_act_proj.{kernel,bias}` silently dropped, and
    `evidence["loaded"] is True`: the exact D37/D39 pathology this row exists to close.

    RED before the fix (measured through `cio.load_checkpoint_into`): no raise, evidence
    `{'step': 100, 'loaded': True}`, zero vec_act_proj leaves in the target after the load."""
    tmp = tempfile.mkdtemp(prefix="test889_arch_hetero_")
    try:
        run_dir = os.path.join(tmp, "ckpt_hetero")
        vec = _tiny_trainer(vector_action=True, seed=0)
        vec.state = vec.state.replace(step_count=100)
        vec.save_checkpoint(run_dir, overwrite=True)
        scalar = _tiny_trainer(vector_action=False, seed=7)
        scalar.state = scalar.state.replace(step_count=200)
        scalar.save_checkpoint(run_dir, overwrite=True)

        leaves = sorted(d for d in os.listdir(run_dir) if d.startswith("checkpoint_"))
        _check("setup: the run dir really holds TWO leaves of DIFFERENT architectures",
              leaves == ["checkpoint_100", "checkpoint_200"], f"leaves={leaves}")
        _check("setup: the SHARED sidecar now describes only the LATER (scalar) save — this is "
              "why an older leaf cannot be verified from it",
              _read_meta(run_dir).get("vector_action") is False
              and _read_meta(run_dir).get("step_count") == 200, str(_read_meta(run_dir)))
        _check("fix pass: save_checkpoint wrote a PER-LEAF record for each leaf",
              all(os.path.exists(os.path.join(run_dir, f"trainer_metadata_{s}.json"))
                  for s in (100, 200)), sorted(os.listdir(run_dir)))

        dst = _tiny_trainer(vector_action=False, seed=1)
        params_before = dst.state.params
        try:
            evidence = cio.load_checkpoint_into(dst, os.path.join(run_dir, "checkpoint_100"))
            _check("BLOCKER: loading the vector-action OLDER leaf into a scalar target REFUSES "
                  "(the shared sidecar describes the newer, scalar save)", False,
                  f"did not raise — evidence={evidence}")
        except RuntimeError as e:
            msg = str(e)
            _check("BLOCKER: ... naming the RECORDED value (True)", "vector_action=True" in msg,
                  msg)
            _check("BLOCKER: ... naming the TARGET value (False)", "vector_action=False" in msg,
                  msg)
        _check("BLOCKER: the refusal leaves dst.state untouched",
              dst.state.params is params_before, "params mutated despite refusal")

        # The matching target still loads the older leaf correctly — the fix adds a refusal, it
        # does not make older leaves unloadable.
        ok = _tiny_trainer(vector_action=True, seed=1)
        evidence_ok = cio.load_checkpoint_into(ok, os.path.join(run_dir, "checkpoint_100"))
        _check("BLOCKER: a MATCHING target still loads that same older leaf, verified",
              evidence_ok["step"] == 100
              and evidence_ok["param_fingerprint"] == cio.param_fingerprint(vec.state.params)
              and evidence_ok.get("verification_skipped") is None, str(evidence_ok))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_fixpass_multi_leaf_tampered_older_leaf_refuses():
    """Gate (iii) on a NON-LATEST leaf — falsified with no adversarial setup at all before the
    fix: an ordinary homogeneous `keep=3` run dir, an older leaf's payload swapped, and the
    tamper was INVISIBLE because the shared sidecar's `step_count` named the newer save and the
    fingerprint check was skipped on that basis.

    RED before the fix: `loaded=True`, no refusal. The per-leaf record is what makes the
    comparison possible instead of skippable."""
    tmp = tempfile.mkdtemp(prefix="test889_arch_tamper_")
    try:
        run_dir = os.path.join(tmp, "ckpt_keep3")
        old = _tiny_trainer(seed=0)
        old.state = old.state.replace(step_count=100)
        old.save_checkpoint(run_dir, overwrite=True)
        new = _tiny_trainer(seed=3)
        new.state = new.state.replace(step_count=200)
        new.save_checkpoint(run_dir, overwrite=True)
        fp_old = _read_leaf_meta(run_dir, 100)["param_fingerprint"]

        # The payload of the OLDER leaf changes underneath its own recorded evidence: swapped
        # for a third, differently-seeded checkpoint saved at the same step.
        forged = _tiny_trainer(seed=42)
        forged.state = forged.state.replace(step_count=100)
        forged_dir = os.path.join(tmp, "ckpt_forged")
        forged.save_checkpoint(forged_dir, overwrite=True)
        fp_forged = cio.param_fingerprint(forged.state.params)
        shutil.rmtree(os.path.join(run_dir, "checkpoint_100"))
        shutil.copytree(os.path.join(forged_dir, "checkpoint_100"),
                        os.path.join(run_dir, "checkpoint_100"))
        _check("setup: the tampered payload really differs from what was recorded for that leaf",
              fp_old != fp_forged, f"{fp_old} == {fp_forged}")

        dst = _tiny_trainer(seed=1)
        params_before = dst.state.params
        meta_before = open(os.path.join(run_dir, "trainer_metadata_100.json"), "rb").read()
        try:
            evidence = cio.load_checkpoint_into(dst, os.path.join(run_dir, "checkpoint_100"))
            _check("BLOCKER: a TAMPERED older leaf in an ordinary keep=3 run dir REFUSES",
                  False, f"did not raise — evidence={evidence}")
        except RuntimeError as e:
            msg = str(e)
            _check("BLOCKER: ... naming the RECORDED fingerprint", fp_old in msg, msg)
            _check("BLOCKER: ... naming the RECOMPUTED (tampered) fingerprint", fp_forged in msg,
                  msg)
            _check("BLOCKER: ... and naming the PER-LEAF record it compared against",
                  "trainer_metadata_100.json" in msg, msg)
        _check("BLOCKER: the refusal leaves dst.state untouched",
              dst.state.params is params_before, "params mutated despite refusal")
        _check("BLOCKER: the disagreement is never overwritten — the per-leaf record on disk is "
              "byte-identical to before the load attempt",
              open(os.path.join(run_dir, "trainer_metadata_100.json"), "rb").read()
              == meta_before, "the per-leaf record was rewritten by a load-time refusal")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_fixpass_legacy_multi_leaf_cannot_verify_refuses():
    """"Cannot verify" must not resolve to "proceed" (D39). A run dir written BEFORE this task —
    several leaves, one shared sidecar, no per-leaf records — has nothing on disk that records
    what an older leaf's parameters should hash to. That load REFUSES; the escape hatch is the
    same explicit flag every other deliberate mismatch uses, and it marks the evidence."""
    tmp = tempfile.mkdtemp(prefix="test889_arch_legacymulti_")
    try:
        run_dir = os.path.join(tmp, "ckpt_legacy_multi")
        a = _tiny_trainer(seed=0)
        a.state = a.state.replace(step_count=100)
        a.save_checkpoint(run_dir, overwrite=True)
        b = _tiny_trainer(seed=7)
        b.state = b.state.replace(step_count=200)
        b.save_checkpoint(run_dir, overwrite=True)
        _strip_leaf_records(run_dir)          # <- makes it a pre-TASK_889 run dir

        dst = _tiny_trainer(seed=1)
        params_before = dst.state.params
        try:
            cio.load_checkpoint_into(dst, os.path.join(run_dir, "checkpoint_100"))
            _check("fix pass: an UNVERIFIABLE older leaf (no per-leaf record, sidecar describes "
                  "another save) REFUSES rather than skipping the check", False, "did not raise")
        except RuntimeError as e:
            msg = str(e)
            _check("fix pass: ... naming the step actually restored (100)", "step=100" in msg,
                  msg)
            _check("fix pass: ... naming the step the shared sidecar describes (200)",
                  "step=200" in msg, msg)
        _check("fix pass: the refusal leaves dst.state untouched",
              dst.state.params is params_before, "params mutated despite refusal")

        dst2 = _tiny_trainer(seed=1)
        evidence = cio.load_checkpoint_into(dst2, os.path.join(run_dir, "checkpoint_100"),
                                            allow_architecture_mismatch=True)
        _check("fix pass: the explicit flag still permits the unverifiable load",
              evidence["loaded"] is True, str(evidence))
        _check("fix pass: ... and the evidence dict RECORDS that nothing was verified",
              evidence.get("fingerprint_verified") is False
              and isinstance(evidence.get("verification_skipped"), str), str(evidence))

        # The LATEST leaf of that same legacy run dir is still verifiable from the shared
        # sidecar, and still loads — the refusal is scoped to what genuinely cannot be checked.
        dst3 = _tiny_trainer(seed=1)
        ev_latest = cio.load_checkpoint_into(dst3, run_dir)
        _check("fix pass: the LATEST leaf of the same legacy run dir still loads, verified "
              "against the shared sidecar", ev_latest["step"] == 200
              and ev_latest.get("verification_skipped") is None, str(ev_latest))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_fixpass_non_numeric_step_count_refuses_by_discipline():
    """MINOR: `int(ckpt_step_count)` used to run outside every guard, so a non-numeric
    `step_count` died with a bare `ValueError` from int() instead of this row's
    RuntimeError/CheckpointRefusalError discipline. A sidecar whose step cannot be read cannot
    be shown to describe this leaf — fail closed, by the named type."""
    tmp = tempfile.mkdtemp(prefix="test889_arch_badstep_")
    try:
        src = _tiny_trainer(seed=0)
        ckpt = os.path.join(tmp, "ckpt")
        src.save_checkpoint(ckpt, overwrite=True)
        _strip_leaf_records(ckpt)
        meta = _read_meta(ckpt)
        meta["step_count"] = "not-a-number"
        _write_meta(ckpt, meta)

        dst = _tiny_trainer(seed=1)
        try:
            cio.load_checkpoint_into(dst, ckpt)
            _check("MINOR: a non-numeric step_count REFUSES", False, "did not raise")
        except cio.CheckpointRefusalError as e:
            _check("MINOR: ... as a CheckpointRefusalError naming the bad value (not a bare "
                  "ValueError from int())", "not-a-number" in str(e), str(e))
        except ValueError as e:
            _check("MINOR: ... as a CheckpointRefusalError, not a bare ValueError", False,
                  f"{type(e).__name__}: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_fixpass_stray_record_never_fails_an_already_valid_save():
    """FIX PASS 3 (MAJOR): the per-leaf ORPHAN SWEEP could kill a training run AFTER a valid save.

    The sweep guarded with `stem.isdigit()` and then called `int(stem)`. `str.isdigit()` is True
    for Unicode digit characters `int()` cannot parse — `'²'.isdigit()` is True, `int('²')`
    raises ValueError — and `except OSError` does not catch ValueError. So a file named
    `trainer_metadata_².json` sitting in the run dir made `save_checkpoint` RAISE, at a point
    where `checkpoint_0`, `trainer_metadata.json` and `trainer_metadata_0.json` were all already
    durably on disk.

    That is not a cosmetic escape: `train_jax_fast.py::_attempt_checkpoint_save` forgives only
    ValueErrors whose text contains "already exists" and re-raises everything else, so this took
    down a training run whose checkpoint was already valid and loadable — flatly contradicting
    the sweep's own contract ("a failure to tidy is never a reason to fail a save that is already
    verified on disk").

    The fix is strictness plus a wider catch, NOT deletion of the stray: a name this function
    never writes is not this function's to remove."""
    tmp = tempfile.mkdtemp(prefix="test889_arch_sweep_")
    try:
        stray_stem = "²"                                    # SUPERSCRIPT TWO
        _check("MAJOR precondition: the stray stem passes str.isdigit() but int() cannot parse "
               "it (the exact isdigit/int gap)",
               stray_stem.isdigit() and not stray_stem.isascii(), repr(stray_stem))

        trainer = _tiny_trainer(vector_action=True, use_prediction_head=False, seed=0)
        ckpt = os.path.join(tmp, "ckpt")
        os.makedirs(ckpt)
        stray = os.path.join(ckpt, f"trainer_metadata_{stray_stem}.json")
        with open(stray, "w") as fh:
            fh.write("{}")
        # A GENUINE orphan too, so the strict stem test is proven not to have broken the sweep's
        # actual job: an ASCII-digit record whose `checkpoint_999` leaf does not exist.
        orphan = os.path.join(ckpt, "trainer_metadata_999.json")
        with open(orphan, "w") as fh:
            fh.write("{}")

        raised = None
        try:
            trainer.save_checkpoint(ckpt, overwrite=True)
        except Exception as e:                        # noqa: BLE001 — the defect is ANY raise
            raised = e
        _check("MAJOR: the save SUCCEEDS with a stray unicode-digit record in the run dir",
               raised is None, f"{type(raised).__name__}: {raised}")

        for name in ("checkpoint_0", "trainer_metadata.json", "trainer_metadata_0.json"):
            _check(f"MAJOR: ... and {name} is on disk (these were already durable when the old "
                   f"sweep raised)", os.path.exists(os.path.join(ckpt, name)),
                   os.path.join(ckpt, name))
        _check("MAJOR: ... and the checkpoint LOADS — the save that used to die was already "
               "producing a valid, loadable artifact",
               cio.load_checkpoint_into(
                   _tiny_trainer(vector_action=True, use_prediction_head=False, seed=5),
                   ckpt)["loaded"] is True, "the checkpoint written past the stray does not load")
        _check("MAJOR: the stray is LEFT ALONE — a name this function never writes is not this "
               "function's to delete", os.path.exists(stray), stray)
        _check("MAJOR: ... while a GENUINE orphan (ASCII stem, no checkpoint_999) is still "
               "swept, so the strict stem test did not break the sweep",
               not os.path.exists(orphan), orphan)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_allow_architecture_mismatch_flag_permits_deliberate_load():
    """Deliberate mismatch stays available ONLY behind the explicit flag — threaded through
    checkpoint_io.load_checkpoint_into exactly like allow_fresh_init already is."""
    tmp = tempfile.mkdtemp(prefix="test889_arch_allow_")
    try:
        src = _tiny_trainer(vector_action=True, use_prediction_head=False, seed=1)
        ckpt = os.path.join(tmp, "ckpt")
        src.save_checkpoint(ckpt, overwrite=True)

        dst = _tiny_trainer(vector_action=False, use_prediction_head=False, seed=0)
        # Without the flag: refuses (already covered above); WITH it: proceeds.
        evidence = cio.load_checkpoint_into(dst, ckpt, allow_architecture_mismatch=True)
        _check("allow_architecture_mismatch=True permits a deliberately mismatched load "
              "instead of refusing", evidence["loaded"] is True, str(evidence))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_gate_vi_matching_load_is_byte_identical_to_saved_params():
    """gate (vi): byte-identity on every scored field wherever the checkpoint loads correctly —
    a matching load must reproduce the saved params EXACTLY, proving the new save/load checks
    add refusals without altering any actual value along the successful path."""
    tmp = tempfile.mkdtemp(prefix="test889_arch_gate6_")
    try:
        src = _tiny_trainer(vector_action=True, use_prediction_head=True, seed=7)
        ckpt = os.path.join(tmp, "ckpt")
        src.save_checkpoint(ckpt, overwrite=True)
        src_leaves = _leaf_dict(src.state.params)

        dst = _tiny_trainer(vector_action=True, use_prediction_head=True, seed=0)
        cio.load_checkpoint_into(dst, ckpt)
        dst_leaves = _leaf_dict(dst.state.params)

        _check("gate (vi): identical leaf sets", set(src_leaves) == set(dst_leaves),
              f"symdiff: {set(src_leaves) ^ set(dst_leaves)}")
        _check("gate (vi): every leaf is EXACTLY byte/value identical (np.array_equal, not "
              "just allclose)",
              all(np.array_equal(src_leaves[k], dst_leaves[k]) for k in src_leaves),
              "at least one leaf differs between saved and loaded params")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    print("[test_architecture_binding_889] gate (i): save records architecture + fingerprint")
    test_gate_i_save_records_architecture_and_fingerprint()
    print("[test_architecture_binding_889] gate (i): matching target loads successfully")
    test_gate_i_matching_target_loads_successfully()
    print("[test_architecture_binding_889] gate (i): mismatched vector_action refuses, naming both")
    test_gate_i_mismatched_vector_action_refuses_naming_both()
    print("[test_architecture_binding_889] gate (ii): prediction-head halt case (constructed)")
    test_gate_ii_prediction_head_enabled_target_vs_false_checkpoint_refuses()
    print("[test_architecture_binding_889] gate (iii): tampered payload -> fingerprint disagreement")
    test_gate_iii_tampered_payload_produces_fingerprint_disagreement()
    print("[test_architecture_binding_889] legacy checkpoint (nothing recorded) does not refuse")
    test_legacy_checkpoint_without_recorded_fields_loads_without_refusing()
    print("[test_architecture_binding_889] allow_architecture_mismatch permits a deliberate load")
    test_allow_architecture_mismatch_flag_permits_deliberate_load()
    print("[test_architecture_binding_889] gate (vi): byte-identity on a matching load")
    test_gate_vi_matching_load_is_byte_identical_to_saved_params()
    print("[test_architecture_binding_889] fix pass: a verified load's evidence gains no keys")
    test_fixpass_verified_load_evidence_carries_no_extra_keys()
    print("[test_architecture_binding_889] fix pass BLOCKER: heterogeneous multi-leaf run dir")
    test_fixpass_multi_leaf_heterogeneous_architecture_refuses()
    print("[test_architecture_binding_889] fix pass BLOCKER: tampered OLDER leaf (keep=3)")
    test_fixpass_multi_leaf_tampered_older_leaf_refuses()
    print("[test_architecture_binding_889] fix pass: legacy multi-leaf 'cannot verify' refuses")
    test_fixpass_legacy_multi_leaf_cannot_verify_refuses()
    print("[test_architecture_binding_889] fix pass MINOR: non-numeric step_count discipline")
    test_fixpass_non_numeric_step_count_refuses_by_discipline()
    print("[test_architecture_binding_889] fix pass MAJOR: a stray record never fails a valid save")
    test_fixpass_stray_record_never_fails_an_already_valid_save()

    print("=" * 84)
    if failures:
        print(f"[test_architecture_binding_889] FAIL — {len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[test_architecture_binding_889] PASS — architecture binding refuses on disagreement, "
          "names both values, never mutates state on refusal, and is byte-identical on a "
          "correct load.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
