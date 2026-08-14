#!/usr/bin/env python3
"""TASK_889 (D39 scope 4) — regression suite for ``tools/stamp_checkpoint_fingerprints.py``.

Covers:
  * **M1** (inherited from the TASK_888 draft of this tool) — the derived provenance name
    MISSES the most-cited production checkpoint (``runs/832_e5/ckpt_imprint_full`` derives
    ``frame_provenance_imprint_full.json``, which does not exist; the real document is
    ``frame_provenance_imprint.json``, whose own ``"checkpoint"`` field names that run dir). The
    sibling fallback must find it, and must REFUSE rather than guess when two documents both
    claim the same checkpoint.
  * **M2** (inherited) — the read-merge-write must not silently RE-BIND an existing provenance
    document to the wrong weights, because ``provenance_path_for`` is non-injective
    (``.../ckpt_foo`` and ``.../foo`` derive the same file).
  * **B2 (TASK_889 / D39, this task's own scope; INVERTED in the fix pass)** — ``vector_action``
    is never a hardcode AND never a silent default: it is RECORDED, else INFERRED from the
    checkpoint's own parameters on disk, else explicitly ASSERTED, else REFUSED. An earlier
    revision of this file asserted that a silently-applied wrong default was intended
    behaviour; that assertion is now inverted.
  * **B2, other half** — a recorded-vs-recomputed ``param_fingerprint`` DISAGREEMENT refuses and
    writes NOTHING, in either sidecar, instead of silently overwriting it (the defect the prior
    TASK_888 draft of this tool shipped).
  * **gate (iv)** — idempotent, and a no-op without ``--apply``.

Every checkpoint here is a throwaway tiny (n_assets=2) model in a tempdir. NOTHING under
``runs/`` is read for writing and ``--apply`` is never pointed at it: the whole point of these
fixtures is that the irreplaceable live checkpoints stay untouched. CPU-only, no GPU, no
``trading_data.db``, no fixture DB.

Run: ``python tools/test_stamp_checkpoint_fingerprints_889.py`` (exit 0 = PASS).
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import contextlib
import hashlib
import importlib.util
import io
import json
import shutil
import sys
import tempfile

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import jax
import jax.numpy as jnp

from mcts_trader.training.config import TrainingConfig
from mcts_trader.training.setup import setup_model_and_trainer
from mcts_trader.jax_agent import checkpoint_io

failures = []
skips = []


def _check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        failures.append(f"{name}: {detail}")


def _skip(name, why):
    """A check group that could not RUN on this host (not one that passed). Recorded and
    reprinted in the summary, never folded silently into the PASS line."""
    print(f"  [SKIP] {name} — {why}")
    skips.append(f"{name}: {why}")


def _stat_pin(path):
    """``(size, mtime_ns)`` — enough to prove a read-only test wrote nothing, without hashing a
    file the test has no business rewriting."""
    st = os.stat(path)
    return st.st_size, st.st_mtime_ns


def _load_stamp_module():
    spec = importlib.util.spec_from_file_location(
        "stamp_889", os.path.join(_REPO, "tools", "stamp_checkpoint_fingerprints.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_checkpoint(run_dir, seed=0, n_assets=2, vector_action=True):
    """A real, loadable checkpoint (with the trainer_metadata.json the stamp tool needs to
    reconstruct an architecture without guessing) at the smallest size this program builds.
    Returns the trainer, so callers can compute the checkpoint's OWN fingerprint independently."""
    cfg = TrainingConfig(name="test_stamp_889", model_size="small", vector_action=vector_action,
                         seed=seed)
    _model, trainer = setup_model_and_trainer(cfg, n_assets)
    trainer.init_train_state(jax.random.PRNGKey(seed), jnp.ones((1, 27, n_assets, n_assets)))
    trainer.save_checkpoint(run_dir, overwrite=True)
    return trainer


def _write_provenance(path, checkpoint_field, **extra):
    doc = {"frame_symbols": ["A/USD", "B/USD"], "n_assets": 2, "source": "test fixture",
           "train_window": ["2021-01-01", "2021-02-01"], "db": "trading_data.db",
           "checkpoint": checkpoint_field}
    doc.update(extra)
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=2)
    return doc


def _sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _read_meta(run_dir):
    with open(os.path.join(run_dir, "trainer_metadata.json")) as fh:
        return json.load(fh)


def test_m1_sibling_fallback_and_ambiguity(st):
    """M1: the 832_e5 shape exactly — a checkpoint whose DERIVED provenance name does not exist
    beside a document that names it."""
    tmp = tempfile.mkdtemp(prefix="test889_stamp_m1_")
    try:
        parent = os.path.join(tmp, "832_e5_sim")
        os.makedirs(parent)
        ckpt = os.path.join(parent, "ckpt_imprint_full")
        _tiny_checkpoint(ckpt, seed=0)
        derived = checkpoint_io.provenance_path_for(ckpt)
        _check("M1 precondition: the DERIVED provenance name does not exist (this is the gap)",
               os.path.basename(derived) == "frame_provenance_imprint_full.json"
               and not os.path.exists(derived), derived)
        real_prov = os.path.join(parent, "frame_provenance_imprint.json")
        _write_provenance(real_prov, ckpt)

        result = st.stamp_one(ckpt, n_assets=2, vector_action_assert=None, apply=True)
        _check("M1: the stamp BINDS into the document that names this checkpoint",
               result.get("provenance") and
               os.path.basename(result["provenance"]) == "frame_provenance_imprint.json",
               str(result))
        with open(real_prov) as fh:
            written = json.load(fh)
        _check("M1: the provenance document now carries the loaded params' fingerprint",
               written.get(checkpoint_io.PROVENANCE_FINGERPRINT_FIELD)
               == result["param_fingerprint"], str(written.get("param_fingerprint")))
        _check("M1: and the document still describes the SAME checkpoint (nothing else "
               "rewritten)", written.get("checkpoint") == ckpt, str(written.get("checkpoint")))

        # gate (iv): idempotent — a second --apply recomputes the same deterministic value and
        # writes nothing.
        before = _sha256(real_prov)
        result2 = st.stamp_one(ckpt, n_assets=2, vector_action_assert=None, apply=True)
        _check("gate (iv): the re-run is idempotent (nothing to write)",
               result2.get("unchanged") is True and result2["wrote"] == [], str(result2))
        _check("gate (iv): and the document's bytes are unchanged", _sha256(real_prov) == before)

        # ambiguity: a SECOND sibling document claiming the same checkpoint is a refusal, never
        # a pick.
        _write_provenance(os.path.join(parent, "frame_provenance_alt.json"), ckpt)
        try:
            st.stamp_one(ckpt, n_assets=2, vector_action_assert=None, apply=False)
            _check("M1: TWO documents claiming one checkpoint is REFUSED", False,
                   "did not raise")
        except RuntimeError as e:
            _check("M1: TWO documents claiming one checkpoint is REFUSED, naming both",
                   "AMBIGUOUS" in str(e) and "frame_provenance_alt.json" in str(e)
                   and "frame_provenance_imprint.json" in str(e), str(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_m2_refuses_rebinding_another_checkpoints_document(st):
    """M2, reproduced: ``provenance_path_for`` is documented non-injective — ``.../ckpt_foo`` and
    ``.../foo`` derive the SAME ``frame_provenance_foo.json``."""
    tmp = tempfile.mkdtemp(prefix="test889_stamp_m2_")
    try:
        parent = os.path.join(tmp, "collide")
        os.makedirs(parent)
        owner = os.path.join(parent, "ckpt_foo")       # the checkpoint the document describes
        squatter = os.path.join(parent, "foo")         # an UNRELATED checkpoint, same derived doc
        _tiny_checkpoint(owner, seed=0)
        _tiny_checkpoint(squatter, seed=7)
        prov = os.path.join(parent, "frame_provenance_foo.json")
        _check("M2 precondition: both checkpoints derive the SAME provenance path",
               checkpoint_io.provenance_path_for(owner)
               == checkpoint_io.provenance_path_for(squatter) == prov, prov)
        _write_provenance(prov, owner)

        # control: the OWNER binds normally.
        owner_result = st.stamp_one(owner, n_assets=2, vector_action_assert=None, apply=True)
        with open(prov) as fh:
            owner_fp = json.load(fh).get(checkpoint_io.PROVENANCE_FINGERPRINT_FIELD)
        _check("M2 control: the checkpoint the document NAMES binds normally",
               owner_fp == owner_result["param_fingerprint"], str(owner_result))

        before = _sha256(prov)
        try:
            st.stamp_one(squatter, n_assets=2, vector_action_assert=None, apply=True)
            _check("M2: stamping an UNRELATED checkpoint into that document is REFUSED",
                   False, "did not raise — the document was re-bound silently")
        except RuntimeError as e:
            _check("M2: stamping an UNRELATED checkpoint into that document is REFUSED, "
                   "naming the mismatch",
                   "REFUSING to stamp" in str(e) and "ckpt_foo" in str(e), str(e))
        _check("M2: the document's bytes are untouched by the refusal (the first checkpoint's "
               "fingerprint survives)", _sha256(prov) == before)
        with open(prov) as fh:
            after = json.load(fh)
        _check("M2: and it still carries the OWNER's fingerprint, not the squatter's",
               after.get(checkpoint_io.PROVENANCE_FINGERPRINT_FIELD) == owner_fp,
               str(after.get(checkpoint_io.PROVENANCE_FINGERPRINT_FIELD)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_b2_vector_action_is_inferred_never_guessed(st):
    """TASK_889 / D39 scope 4, FIX PASS — and the INVERSION of what this test used to assert.

    The first draft hardcoded ``vector_action=True`` inside ``_build_compatible_trainer``. The
    second moved it into a named ``DEFAULT_VECTOR_ACTION`` applied SILENTLY when nothing was
    recorded — and this test asserted that silent acceptance of a wrong guess as INTENDED
    behaviour ("SILENTLY ACCEPTED as a DIFFERENT fingerprint"). A silently-applied default IS
    the hardcode: it still writes a well-formed WRONG fingerprint into the document TASK_886
    certifies against. That assertion is inverted here.

    The architecture is now DERIVED, in order: (1) the checkpoint's own recorded value;
    (2) INFERENCE from its parameters on disk — a target-less orbax read answers whether
    ``vector_action_embedding.vec_act_proj`` is in the payload, which IS the architecture;
    (3) only then an explicit operator assertion. Nothing recorded and nothing inferable is a
    REFUSAL. Every disagreement between two available sources is a refusal too."""
    tmp = tempfile.mkdtemp(prefix="test889_stamp_b2_")
    try:
        ckpt = os.path.join(tmp, "ckpt_scalar")
        trainer_src = _tiny_checkpoint(ckpt, seed=0, vector_action=False)
        fp_src = checkpoint_io.param_fingerprint(trainer_src.state.params)

        meta = _read_meta(ckpt)
        _check("precondition: trainer.save_checkpoint records vector_action=False "
               "(TASK_889 scope 1)", meta.get("vector_action") is False, str(meta))

        # (1) the RECORDED value wins, with nothing asserted on the command line.
        result = st.stamp_one(ckpt, n_assets=2, vector_action_assert=None, apply=False)
        _check("B2: the checkpoint's RECORDED vector_action=False is used, with no operator "
               "assertion and no default anywhere", result["vector_action"] is False,
               str(result))
        _check("B2: ... and the recomputed fingerprint matches the checkpoint's OWN fingerprint "
               "(a wrong architecture would produce a DIFFERENT one — 2 extra leaves)",
               result["param_fingerprint"] == fp_src, str(result))
        _check("B2: ... and the result NAMES where the value came from",
               "recorded" in result["vector_action_source"], str(result))

        # (2) NOTHING recorded — the pre-TASK_889 shape. The old code applied its True default
        # here and accepted the resulting wrong fingerprint in silence. It now INFERS False from
        # the parameters on disk and gets the RIGHT answer with nothing asserted.
        field = checkpoint_io.PROVENANCE_FINGERPRINT_FIELD
        virgin_meta = {k: v for k, v in meta.items() if k not in ("vector_action", field)}
        with open(os.path.join(ckpt, "trainer_metadata.json"), "w") as fh:
            json.dump(virgin_meta, fh)

        inferred, why = st.infer_vector_action(os.path.join(ckpt, "checkpoint_0"))
        _check("B2: infer_vector_action reads the architecture off the SCALAR checkpoint's own "
               "params (vector_action_embedding ABSENT)", inferred is False, f"{inferred}: {why}")

        result_inferred = st.stamp_one(ckpt, n_assets=2, vector_action_assert=None, apply=False)
        _check("B2 (INVERTED): with NOTHING recorded, the value is INFERRED from the "
               "checkpoint's own parameters — not defaulted to True",
               result_inferred["vector_action"] is False, str(result_inferred))
        _check("B2 (INVERTED): ... so the recomputed fingerprint is the CHECKPOINT's OWN, where "
               "the silently-applied default produced a well-formed WRONG one",
               result_inferred["param_fingerprint"] == fp_src, str(result_inferred))
        _check("B2 (INVERTED): ... and the result says it was INFERRED",
               "INFERRED" in result_inferred["vector_action_source"], str(result_inferred))

        # (3) an operator assertion that CONTRADICTS the checkpoint is a REFUSAL, not an
        # override — this is the wrong guess the old code accepted, now caught even with
        # NOTHING recorded in the metadata sidecar.
        try:
            st.stamp_one(ckpt, n_assets=2, vector_action_assert=True, apply=False)
            _check("B2 (INVERTED): asserting the WRONG architecture (--vector-action for a "
                   "scalar checkpoint) is REFUSED, not silently accepted", False,
                   "did not raise — a wrong assertion was accepted")
        except RuntimeError as e:
            _check("B2 (INVERTED): ... naming both the assertion and the checkpoint's answer",
                   "vector-action" in str(e) and "vector_action=False" in str(e), str(e))

        # ... and asserting the value the checkpoint agrees with is harmless.
        result_agree = st.stamp_one(ckpt, n_assets=2, vector_action_assert=False, apply=False)
        _check("B2: an assertion that AGREES with the checkpoint still works",
               result_agree["param_fingerprint"] == fp_src, str(result_agree))

        # (4) a checkpoint whose RECORD contradicts its own PAYLOAD refuses — the record is not
        # trusted blindly either.
        lying_meta = {**virgin_meta, "vector_action": True}
        with open(os.path.join(ckpt, "trainer_metadata.json"), "w") as fh:
            json.dump(lying_meta, fh)
        try:
            st.stamp_one(ckpt, n_assets=2, vector_action_assert=None, apply=False)
            _check("B2: a RECORDED vector_action that contradicts the checkpoint's own params "
                   "is REFUSED", False, "did not raise")
        except RuntimeError as e:
            _check("B2: ... naming the record and the payload",
                   "records vector_action=True" in str(e) and "ABSENT" in str(e), str(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_b2_uninferable_architecture_refuses_and_demands_an_assertion(st):
    """The last leg of "never a guess": when the architecture is neither recorded NOR inferable,
    the tool REFUSES and says which flag to pass — it does not fall back to a default. Exercised
    against ``resolve_vector_action`` with inference stubbed to "cannot answer", because that is
    the exact contract under test (an unreadable/foreign payload is the real-world cause, and
    constructing one would test orbax's error handling instead of this decision)."""
    real_infer = st.infer_vector_action
    st.infer_vector_action = lambda leaf: (None, "the payload could not be read target-less")
    try:
        try:
            st.resolve_vector_action({}, "/nonexistent/leaf", None, "ckpt_x")
            _check("B2: an UNINFERABLE, unrecorded architecture is REFUSED, never defaulted",
                   False, "did not raise")
        except RuntimeError as e:
            _check("B2: ... naming the refusal and the flags that would resolve it",
                   "UNKNOWN" in str(e) and "--vector-action" in str(e)
                   and "--no-vector-action" in str(e), str(e))
        value, source = st.resolve_vector_action({}, "/nonexistent/leaf", False, "ckpt_x")
        _check("B2: ... and an EXPLICIT assertion is what unblocks it (last resort, disclosed)",
               value is False and "ASSERTED" in source, f"{value} / {source}")
    finally:
        st.infer_vector_action = real_infer


def test_b2_vector_action_inference_sees_a_vector_checkpoint(st):
    """The positive half of the inference: a VECTOR-action checkpoint is read as such off its
    own payload (the negative half — a scalar one — is covered above). Ground truth is the
    presence of ``vector_action_embedding.vec_act_proj`` in the saved params."""
    tmp = tempfile.mkdtemp(prefix="test889_stamp_infer_")
    try:
        ckpt = os.path.join(tmp, "ckpt_vector")
        trainer_src = _tiny_checkpoint(ckpt, seed=0, vector_action=True)
        fp_src = checkpoint_io.param_fingerprint(trainer_src.state.params)
        inferred, why = st.infer_vector_action(os.path.join(ckpt, "checkpoint_0"))
        _check("B2: infer_vector_action reads TRUE off a vector-action checkpoint's params",
               inferred is True, f"{inferred}: {why}")

        # strip the record entirely: inference alone must reproduce the right fingerprint
        meta = _read_meta(ckpt)
        field = checkpoint_io.PROVENANCE_FINGERPRINT_FIELD
        with open(os.path.join(ckpt, "trainer_metadata.json"), "w") as fh:
            json.dump({k: v for k, v in meta.items() if k not in ("vector_action", field)}, fh)
        result = st.stamp_one(ckpt, n_assets=2, vector_action_assert=None, apply=False)
        _check("B2: an unrecorded VECTOR-action checkpoint stamps the right fingerprint from "
               "inference alone", result["vector_action"] is True
               and result["param_fingerprint"] == fp_src, str(result))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_fixpass_real_cross_device_checkpoint_infers_without_deserializing(st):
    """FIX PASS 3 — the BLOCKER: the tool crashed on one of its OWN default targets, and the
    synthetic fixtures in this file are structurally incapable of catching it.

    ``runs/832_e5/ckpt_imprint_full`` was saved on a GPU host: its ``checkpoint_2000/_sharding``
    records ``device_str: "cuda:0"`` where the other six survivors record ``TFRT_CPU_0``. The
    cycle-2 ``infer_vector_action`` answered the architecture question with a target-less
    ``flax_checkpoints.restore_checkpoint(target=None)``, which DESERIALIZES every array and so
    depends on that saved sharding — on this CPU-only workstation it raised ``ValueError:
    sharding passed to deserialization should be specified, concrete and an instance of
    jax.sharding.Sharding. Got None``. The tool then REFUSED a KNOWABLE architecture and (before
    the batch fix) that refusal killed the whole 7-member default run at member 7, discarding six
    already-computed verdicts.

    Nothing about that checkpoint is ambiguous: its ``_METADATA`` plainly lists
    ``('params', 'vector_action_embedding', 'vec_act_proj', ...)`` — ``vector_action=True``, the
    same as the other six — and the production loader restores it fine given a correctly built
    target.

    WHY THE OTHER TESTS CANNOT COVER THIS. Every other checkpoint in this suite is written by
    THIS process on THIS host, so its sharding is always local and the cross-device read never
    happens; and ``test_b2_uninferable_architecture_refuses_and_demands_an_assertion``
    monkeypatches ``infer_vector_action`` away entirely. Only the real artifact exercises it.

    READ-ONLY, ALWAYS. This runs the inference/resolution path only, never ``stamp_one`` and
    never ``--apply``; it pins the size+mtime of the two real sidecars beside the checkpoint and
    asserts they are untouched."""
    ckpt = os.path.join(_REPO, "runs", "832_e5", "ckpt_imprint_full")
    leaf = os.path.join(ckpt, "checkpoint_2000")
    meta_path = os.path.join(ckpt, "trainer_metadata.json")
    prov_path = os.path.join(_REPO, "runs", "832_e5", "frame_provenance_imprint.json")
    if not (os.path.isdir(leaf) and os.path.exists(meta_path)):
        _skip("fix-pass: real cross-device checkpoint inference",
              f"{leaf} is not present on this host (the real 832_e5 checkpoint lives on the "
              f"workstation; nothing is asserted about a checkpoint that is not here)")
        return
    pins = {p: _stat_pin(p) for p in (meta_path, prov_path) if os.path.exists(p)}

    sharding_path = os.path.join(leaf, "_sharding")
    if os.path.exists(sharding_path):
        with open(sharding_path) as fh:
            sharding_raw = fh.read()
        _check("fix-pass precondition: this leaf's saved sharding names a NON-LOCAL device "
               "(cuda:0) — the exact condition a target-less array read cannot concretize here",
               "cuda:0" in sharding_raw,
               "the leaf no longer records a cuda device; this test's premise has changed")

    inferred, why = st.infer_vector_action(leaf)
    _check("fix-pass BLOCKER: the REAL cross-device checkpoint's architecture is INFERRED "
           "(True), not refused as unknowable", inferred is True, f"{inferred}: {why}")
    _check("fix-pass BLOCKER: ... and the reason names the STRUCTURE it read, not an array read",
           inferred is True and st.ORBAX_TREE_METADATA in why
           and st.VECTOR_ACTION_LEAF_PARENT in why, str(why))

    meta = _read_meta(ckpt)
    _check("fix-pass precondition: this checkpoint records NO vector_action, so inference is "
           "the only source that can answer (recorded is absent, nothing is asserted)",
           meta.get("vector_action") is None, str(meta.get("vector_action")))
    # Caught, not allowed to propagate: a regression here must be reported as a FAILED CHECK with
    # the refusal quoted, not as a traceback that takes the rest of this suite down with it.
    try:
        value, source = st.resolve_vector_action(meta, leaf, None,
                                                 "runs/832_e5/ckpt_imprint_full")
    except RuntimeError as e:
        value, source = None, f"REFUSED: {e}"
    _check("fix-pass BLOCKER: the full precedence resolves it to True with NOTHING asserted on "
           "the command line", value is True, f"{value} / {source}")
    _check("fix-pass BLOCKER: ... and says the value was INFERRED",
           value is True and "INFERRED" in source, str(source))

    for path, before in pins.items():
        _check(f"fix-pass: read-only — {os.path.basename(path)} is untouched (size+mtime)",
               _stat_pin(path) == before, f"{before} -> {_stat_pin(path)}")


def test_fixpass_one_refusal_does_not_abort_the_batch(st):
    """FIX PASS 3, the other half of the BLOCKER: ``main()``'s per-checkpoint loop had no
    try/except, so ONE member's refusal aborted the whole batch — the operator learned nothing
    about the members that were never reached, and re-running to find out costs a full load of
    every earlier checkpoint.

    Constructed with the refusal this tool is most likely to raise in the field (a recorded-vs-
    recomputed fingerprint disagreement), placed FIRST so the assertion is genuinely "the run
    continued past it". Dry run: nothing is written for either member."""
    tmp = tempfile.mkdtemp(prefix="test889_stamp_batch_")
    try:
        parent = os.path.join(tmp, "batch")
        os.makedirs(parent)
        bad = os.path.join(parent, "ckpt_bad")
        good = os.path.join(parent, "ckpt_good")
        _tiny_checkpoint(bad, seed=0)
        _tiny_checkpoint(good, seed=1)
        bad_meta = os.path.join(bad, "trainer_metadata.json")
        with open(bad_meta) as fh:
            tampered = json.load(fh)
        tampered[checkpoint_io.PROVENANCE_FINGERPRINT_FIELD] = "0" * 64
        with open(bad_meta, "w") as fh:
            json.dump(tampered, fh)
        good_meta = os.path.join(good, "trainer_metadata.json")
        bad_sha, good_sha = _sha256(bad_meta), _sha256(good_meta)

        argv = sys.argv
        sys.argv = ["stamp_checkpoint_fingerprints.py", "--n-assets", "2",
                    "--checkpoint", bad, "--checkpoint", good]
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                rc = st.main()
        finally:
            sys.argv = argv
        out = buf.getvalue()

        _check("batch: the refusing member is CLASSIFIED (REFUSED), not raised out of main()",
               "REFUSED:" in out and "0" * 64 in out, out[-1500:])
        _check("batch: the member AFTER the refusal is still processed",
               "WOULD stamp" in out and out.index("REFUSED:") < out.index("WOULD stamp"),
               out[-1500:])
        _check("batch: main() exits NON-ZERO when any member refused (no script can read "
               "'finished' as 'all clean')", rc == 1, str(rc))
        _check("batch: the summary counts the refusal separately from a skip",
               "1 REFUSED" in out, out[-800:])
        _check("batch: the refusal body is not clipped down to the 300-char path budget "
               "(the operator instruction survives)", "Nothing written" in out, out[-1500:])
        _check("batch: nothing was written for the refusing member",
               _sha256(bad_meta) == bad_sha, "the refused member's sidecar was rewritten")
        _check("batch: nothing was written for the surviving member either (dry run)",
               _sha256(good_meta) == good_sha, "a dry run wrote a sidecar")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_b2_fingerprint_disagreement_refuses_and_writes_nothing(st):
    """TASK_889 / D39 B2, other half: a recorded-vs-recomputed DISAGREEMENT — precisely the "a
    checkpoint changed underneath you" signal this tool exists to produce — REFUSES and writes
    NOTHING, instead of the prior draft's silent overwrite."""
    tmp = tempfile.mkdtemp(prefix="test889_stamp_b2fp_")
    try:
        parent = os.path.join(tmp, "run")
        os.makedirs(parent)
        ckpt = os.path.join(parent, "ckpt_x")
        _tiny_checkpoint(ckpt, seed=0)
        prov_path = os.path.join(parent, "frame_provenance_x.json")
        _write_provenance(prov_path, ckpt)

        # First stamp: real, clean write. trainer_metadata.json is ALREADY correct (TASK_889
        # scope 1/2 write it at save time), so only the hand-built provenance fixture (which
        # never carries a fingerprint) needs one.
        result = st.stamp_one(ckpt, n_assets=2, vector_action_assert=None, apply=True)
        _check("setup: first stamp writes the provenance sidecar (metadata is already correct "
               "from save time)", result["wrote"] == ["frame_provenance_x.json"], str(result))
        meta_path = os.path.join(ckpt, "trainer_metadata.json")
        prov_before = _sha256(prov_path)

        # Corrupt the RECORDED metadata fingerprint (simulating drift/tampering) — never the
        # weights themselves; this tool never opens the orbax leaf for writing.
        with open(meta_path) as fh:
            tampered = json.load(fh)
        real_fp = tampered[checkpoint_io.PROVENANCE_FINGERPRINT_FIELD]
        tampered[checkpoint_io.PROVENANCE_FINGERPRINT_FIELD] = "0" * 64
        with open(meta_path, "w") as fh:
            json.dump(tampered, fh)
        tampered_sha = _sha256(meta_path)

        try:
            st.stamp_one(ckpt, n_assets=2, vector_action_assert=None, apply=True)
            _check("B2: a recorded-vs-recomputed metadata fingerprint DISAGREEMENT refuses",
                   False, "did not raise — silently overwrote the disagreement")
        except RuntimeError as e:
            _check("B2: ... and names BOTH values (the tampered record and the recomputed one)",
                   "0" * 64 in str(e) and real_fp in str(e), str(e))
        _check("B2: the tampered metadata file is UNCHANGED by the refusal (never overwritten)",
               _sha256(meta_path) == tampered_sha, "metadata was rewritten despite the refusal")
        _check("B2: the untouched provenance document is ALSO unchanged (writes nothing means "
               "nothing, not just the offending file)",
               _sha256(prov_path) == prov_before, "provenance was rewritten despite the refusal")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_b2_provenance_fingerprint_disagreement_refuses(st):
    """TASK_889 / D39 B2: the SAME disagreement check, on the provenance document's own
    recorded fingerprint rather than the metadata sidecar's."""
    tmp = tempfile.mkdtemp(prefix="test889_stamp_b2prov_")
    try:
        parent = os.path.join(tmp, "run")
        os.makedirs(parent)
        ckpt = os.path.join(parent, "ckpt_y")
        _tiny_checkpoint(ckpt, seed=0)
        prov_path = os.path.join(parent, "frame_provenance_y.json")
        _write_provenance(prov_path, ckpt)

        st.stamp_one(ckpt, n_assets=2, vector_action_assert=None, apply=True)
        meta_path = os.path.join(ckpt, "trainer_metadata.json")
        meta_before = _sha256(meta_path)

        with open(prov_path) as fh:
            prov_doc = json.load(fh)
        prov_doc[checkpoint_io.PROVENANCE_FINGERPRINT_FIELD] = "f" * 64
        with open(prov_path, "w") as fh:
            json.dump(prov_doc, fh)
        prov_tampered_sha = _sha256(prov_path)

        try:
            st.stamp_one(ckpt, n_assets=2, vector_action_assert=None, apply=True)
            _check("B2: a recorded-vs-recomputed PROVENANCE fingerprint disagreement refuses",
                   False, "did not raise")
        except RuntimeError as e:
            _check("B2: ... naming the disagreement", "disagrees" in str(e), str(e))
        _check("B2: the tampered provenance document is UNCHANGED by the refusal",
               _sha256(prov_path) == prov_tampered_sha, "provenance was rewritten")
        _check("B2: the metadata sidecar is ALSO untouched (nothing written for either file)",
               _sha256(meta_path) == meta_before, "metadata was rewritten despite the refusal")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_gate_iv_idempotent_and_noop_without_apply(st):
    """gate (iv): the stamp is a no-op without --apply, and idempotent with it.

    Uses a checkpoint with `vector_action`/`param_fingerprint` stripped (simulating a
    pre-TASK_889 legacy checkpoint) so there is something real for the FIRST --apply to write —
    a freshly `trainer.save_checkpoint`-d one already carries both (TASK_889 scope 1/2), which
    would make this gate trivially "nothing to do" rather than exercising the write path."""
    tmp = tempfile.mkdtemp(prefix="test889_stamp_gate4_")
    try:
        ckpt = os.path.join(tmp, "ckpt_gate4")
        _tiny_checkpoint(ckpt, seed=0)
        meta_path = os.path.join(ckpt, "trainer_metadata.json")
        with open(meta_path) as fh:
            legacy_meta = json.load(fh)
        field = checkpoint_io.PROVENANCE_FINGERPRINT_FIELD
        legacy_meta = {k: v for k, v in legacy_meta.items() if k not in ("vector_action", field)}
        with open(meta_path, "w") as fh:
            json.dump(legacy_meta, fh)
        before_mtime = os.path.getmtime(meta_path)
        before_sha = _sha256(meta_path)

        result_dry = st.stamp_one(ckpt, n_assets=2, vector_action_assert=None, apply=False)
        _check("gate (iv): a dry run reports it WOULD write",
               result_dry["status"] == "would-stamp" and result_dry["wrote"], str(result_dry))
        _check("gate (iv): a dry run is a NO-OP — the file's mtime is untouched",
               os.path.getmtime(meta_path) == before_mtime, "mtime changed without --apply")
        _check("gate (iv): a dry run is a NO-OP — the file's bytes are untouched",
               _sha256(meta_path) == before_sha, "bytes changed without --apply")

        result_apply1 = st.stamp_one(ckpt, n_assets=2, vector_action_assert=None, apply=True)
        _check("gate (iv): --apply actually writes", result_apply1["wrote"], str(result_apply1))
        after_first_sha = _sha256(meta_path)

        result_apply2 = st.stamp_one(ckpt, n_assets=2, vector_action_assert=None, apply=True)
        _check("gate (iv): a second --apply is idempotent (nothing left to write)",
               result_apply2["unchanged"] is True and result_apply2["wrote"] == [],
               str(result_apply2))
        _check("gate (iv): and the file's bytes are IDENTICAL to after the first --apply "
               "(no drift across re-runs)", _sha256(meta_path) == after_first_sha)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    st = _load_stamp_module()
    print("[test_stamp_889] M1: sibling provenance fallback + ambiguity refusal")
    test_m1_sibling_fallback_and_ambiguity(st)
    print("[test_stamp_889] M2: refuse re-binding another checkpoint's provenance document")
    test_m2_refuses_rebinding_another_checkpoints_document(st)
    print("[test_stamp_889] B2: vector_action is INFERRED from the checkpoint, never guessed")
    test_b2_vector_action_is_inferred_never_guessed(st)
    print("[test_stamp_889] B2: an uninferable architecture refuses and demands an assertion")
    test_b2_uninferable_architecture_refuses_and_demands_an_assertion(st)
    print("[test_stamp_889] B2: inference reads a VECTOR-action checkpoint correctly")
    test_b2_vector_action_inference_sees_a_vector_checkpoint(st)
    print("[test_stamp_889] fix pass: the REAL cross-device checkpoint infers without "
          "deserializing")
    test_fixpass_real_cross_device_checkpoint_infers_without_deserializing(st)
    print("[test_stamp_889] fix pass: one member's refusal does not abort the batch")
    test_fixpass_one_refusal_does_not_abort_the_batch(st)
    print("[test_stamp_889] B2: metadata fingerprint disagreement refuses, writes nothing")
    test_b2_fingerprint_disagreement_refuses_and_writes_nothing(st)
    print("[test_stamp_889] B2: provenance fingerprint disagreement refuses, writes nothing")
    test_b2_provenance_fingerprint_disagreement_refuses(st)
    print("[test_stamp_889] gate (iv): idempotent and a no-op without --apply")
    test_gate_iv_idempotent_and_noop_without_apply(st)

    print("=" * 84)
    for s in skips:
        print(f"[test_stamp_889] SKIPPED (not a pass): {s}")
    if failures:
        print(f"[test_stamp_889] FAIL — {len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[test_stamp_889] PASS — the retro-stamp tool never hardcodes vector_action, infers "
          "the architecture from the structure on disk (device-independently), refuses on a "
          "recorded-vs-recomputed disagreement, is idempotent/no-op-safe, and one member's "
          "refusal never aborts the batch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
