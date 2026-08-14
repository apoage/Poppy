#!/usr/bin/env python3
"""TASK_889 (D39 scope 4) — retro-stamp ``vector_action`` + ``param_fingerprint`` into
surviving checkpoints' ``trainer_metadata.json`` AND their frame-provenance documents.

NAMING (operator correction, 2026-08-12, binding): this operation is a **retro-stamp**, never
a "backfill". In this repo *backfill* already means TASK_858's append-only INGEST of historical
minute bars, and *back-fill* in the data sense is a FORBIDDEN look-ahead. This operation touches
NO price data at all — it computes a hash over parameters already on disk and writes it (plus
the architecture that produced it) into JSON sidecars. Naming ambiguity killed four consecutive
cycles on this lane before this rule was stated; the module and every symbol in it says
"stamp", never "backfill".

New checkpoints get ``vector_action`` and ``param_fingerprint`` at SAVE time for free
(``trainer.py::save_checkpoint`` for the metadata sidecar, ``imprint.py::build_frame_provenance``
for the provenance document — TASK_889 scope 1/2). Checkpoints saved BEFORE that change have
neither. This is the one-shot repair: load each surviving checkpoint through the SAME
fail-closed loader every other TASK_888/889 site uses (``checkpoint_io.load_checkpoint_into`` —
resolves via the ONE resolution function, refuses on a hang-risk/foreign-shape/no-op load exactly
like eval_vector.py / train_jax_fast.py / distill.py / real_trade_jax.py), then records the
resulting architecture and fingerprint in both places.

WHY BOTH FILES, AND WHICH ONE THE CONTRACT ACTUALLY READS (fix-pass M5, TASK_888R). TASK_886's
certification contract reads ``universe_pin["provenance_param_fingerprint"]``, which is sourced
from the FRAME-PROVENANCE document (``frame_provenance_<stem>.json``, field
``param_fingerprint``) — NOT from ``trainer_metadata.json``. Writing only the metadata sidecar
would leave ``verified: true`` structurally unreachable. The metadata sidecar is still written
because that is where ``trainer.save_checkpoint`` puts it for every FUTURE checkpoint, and it is
the copy that survives a lost/never-written provenance.

NEVER A HARDCODE, AND NEVER A GUESS (TASK_889 / D39 B2, the defect this row fixes). The first
draft of this tool hardcoded ``vector_action=True`` inside its trainer-construction helper: a
scalar checkpoint was then fingerprinted over a target carrying 2 EXTRA leaves at the tool's own
``PRNGKey(0)`` init, and that wrong value was written into the document TASK_886 certifies
against. The second draft moved the hardcode into a named ``DEFAULT_VECTOR_ACTION`` and applied
it SILENTLY whenever nothing was recorded — which is the same defect wearing a constant's name: a
wrong guess was still accepted without a word and written under ``--apply``.

This version DERIVES the answer from the checkpoint itself, in a strict order, and refuses when
it cannot:

  1. the checkpoint's OWN recorded ``vector_action`` (present once it has been saved — or
     stamped — under TASK_889);
  2. INFERENCE from the STRUCTURE ON DISK (``infer_vector_action``): orbax's own ``_METADATA``
     tree listing inside the leaf names every key path the save wrote, so it answers whether
     ``vector_action_embedding.vec_act_proj.{kernel,bias}`` is in this checkpoint at all — which
     IS the architecture, not a belief about it;
  3. only if BOTH are unavailable, the operator's explicit ``--vector-action`` /
     ``--no-vector-action`` assertion — and with neither recorded, inferable, nor asserted, the
     checkpoint is REFUSED rather than defaulted.

Any DISAGREEMENT between those sources — record vs parameters, or operator assertion vs either —
REFUSES and writes nothing, the same way a fingerprint disagreement does. ``--n-assets`` remains
an assertion (it is recorded nowhere and cannot be inferred from a leaf whose per-asset shapes
are what the assertion is needed to interpret).

RECORDED-VS-RECOMPUTED DISAGREEMENT REFUSES AND WRITES NOTHING (TASK_889 / D39 B2, the other
half). If a checkpoint's sidecar ALREADY carries a ``param_fingerprint`` (e.g. the 7 survivors'
run-32 orphaned values — untrusted, but real bytes on disk) and the freshly recomputed one
disagrees, that is precisely the "a checkpoint changed underneath you — or the assumed
architecture was wrong" signal this tool exists to produce. The prior draft OVERWROTE it
silently; this one RAISES and writes NOTHING for that checkpoint, in EITHER sidecar, dry run or
``--apply`` alike (a disagreement is worth stopping the whole run for, not something ``--apply``
should be needed to discover).

DRY-RUN BY DEFAULT. With no ``--apply`` nothing is written — the run reports what it WOULD
write. ``--apply`` matches this repo's existing freeze/promote convention (``imprint.py
--write-parity-baseline``, ``861/smoke_e2e.py --promote-baselines``). ``--apply`` against
``runs/`` stays PLANNER-GATED — D39: "until architecture is recorded at SAVE time", which
TASK_889 is what makes true.

WHAT THIS DOES NOT DO:
  * mutate any weight/parameter payload — only the two JSON sidecars are written (read,
    add/overwrite keys, atomic write via a ``.tmp`` + ``os.replace``). The orbax
    ``checkpoint_<step>/`` directory is never opened for writing.
  * resurrect a checkpoint that no longer exists. A missing path is reported SKIPPED, not an
    error.
  * guess h_cycles/l_cycles/state_dim. Stamping requires the checkpoint's OWN
    ``trainer_metadata.json`` to already record them (every checkpoint
    ``trainer.save_checkpoint`` has ever written carries these) — a checkpoint missing that
    sidecar entirely is reported SKIPPED with the reason, never guessed at.
  * create a provenance document. An absent one is reported, never synthesized.
  * overwrite a recorded-vs-recomputed disagreement. It REFUSES instead (see above).

IDEMPOTENT: a checkpoint already carrying a recorded ``vector_action``/``param_fingerprint``
that agrees with the recompute needs no write; re-running produces the SAME deterministic
result every time — nothing accumulates, nothing drifts.

ONE MEMBER'S REFUSAL NEVER ABORTS THE BATCH (TASK_889 fix pass). Every refusal above is
deliberate and fail-closed, but a batch tool that DIES on member 7 of 7 leaves the operator with
no idea what the other 6 would have done — and re-running to find out costs a full load of every
earlier checkpoint. A refusal is classified as REFUSED for that checkpoint, printed in full, and
the run continues; the process exits NON-ZERO if any member refused, so no script can read
"finished" as "all clean". Nothing is written for a refused member, exactly as before.

Default targets: the 7 checkpoints named in TASK_888/889 — every load-bearing survivor at the
time this task was written. Pass ``--checkpoint`` (repeatable) to stamp additional/different
paths (e.g. on gpu-server, which this workstation cannot reach).

CPU-only, no GPU. Never touches ``trading_data.db`` or the fixture DB (neither is on the path of
loading a checkpoint's own parameter payload).

Run:
    python tools/stamp_checkpoint_fingerprints.py                 # DRY RUN, the 7 defaults
    python tools/stamp_checkpoint_fingerprints.py --apply         # actually write
    python tools/stamp_checkpoint_fingerprints.py --apply --checkpoint <dir> [--checkpoint ...]
    python tools/stamp_checkpoint_fingerprints.py --n-assets 8 --checkpoint <fixture ckpt>
    # --vector-action / --no-vector-action are a LAST RESORT, for a checkpoint whose
    # architecture is neither recorded nor inferable from its own parameters.
"""
import argparse
import json
import os
import sys
import traceback

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import jax
import jax.numpy as jnp

from mcts_trader.training.config import TrainingConfig
from mcts_trader.training.setup import setup_model_and_trainer
from mcts_trader.jax_agent import checkpoint_io

# "Measured cost: 7 locally" (TASK_888/889, D38/D39): every load-bearing checkpoint that
# survived at the time this task was written. Repo-relative so the script runs from anywhere.
DEFAULT_CHECKPOINTS = [
    "runs/861_ref/ckpt_ref_s0",
    "runs/861_ref/ckpt_ref_s1",
    "runs/861_ref/ckpt_ref_s2",
    "runs/861_imprint/ckpt_s0",
    "runs/861_imprint/ckpt_s1",
    "runs/861_imprint/ckpt_s2",
    "runs/832_e5/ckpt_imprint_full",
]

# Real (non-fixture) battery checkpoints in this repo are trained on the 64-asset universe
# (828 Stage-3 / 861 / 832). n_assets is NOT recorded in trainer_metadata.json (check_checkpoint_
# arch's own documented gap), so this is a default, not a derivation — override with --n-assets
# for a checkpoint trained on a different universe (e.g. an 8-asset fixture checkpoint).
DEFAULT_N_ASSETS = 64

# TASK_889 / D39 B2 (fix pass): there is NO default `vector_action` any more. A named constant
# applied silently when nothing is recorded is the hardcode with a better name — it still writes
# a GUESS into the document TASK_886 certifies against, and the second draft's own regression
# test asserted that silent acceptance as intended behaviour. The value is now RECORDED,
# INFERRED from the parameters on disk, or ASSERTED on the command line — and if none of the
# three is available the checkpoint is REFUSED. See `infer_vector_action` / `resolve_vector_action`.

# The parameter subtree whose PRESENCE is what `vector_action=True` means on disk. Read off the
# canonical model (muzero_model.py builds `vector_action_embedding` only under
# `use_vector_action=True`, and its single Dense is `vec_act_proj`), and verified against both
# architectures on disk: a vector-action checkpoint records
# `('params', 'vector_action_embedding', 'vec_act_proj', {'kernel','bias'})` and a scalar one
# records no `vector_action_embedding` key path at all.
VECTOR_ACTION_SUBTREE = "vector_action_embedding"
VECTOR_ACTION_LEAF_PARENT = "vec_act_proj"

# orbax's own tree-structure listing INSIDE a checkpoint leaf. It names every key path the save
# wrote, as JSON, with no array data — which is exactly what the architecture question needs and
# nothing more. flax's `_is_orbax_checkpoint` treats its presence as one of the three markers of
# an orbax checkpoint, and all 117 `checkpoint_*` leaves in this repo carry one (measured
# 2026-08-13; every one is also ~270KB, three orders under SIDECAR_MAX_BYTES — the listing scales
# with LEAF COUNT, which `model_size` does not change, not with parameter count).
ORBAX_TREE_METADATA = "_METADATA"

# `clip_for_refusal`'s default (300 chars) is sized for ONE interpolated PATH. A refusal BODY
# raised by `stamp_one` is a multi-line operator instruction that names both disagreeing values
# and what to do next; clipping THAT at 300 would cut the instruction off mid-sentence and lose
# information the previous, aborting version printed in full via its traceback. 4000 matches the
# refusal-echo budget TASK_888 settled on (a 1.6M-char echo -> <4k, path still named): the one
# unbounded component (an operator-supplied `--checkpoint`) is still capped.
REFUSAL_BODY_MAX = 4000


def _display_path(p):
    p = os.path.abspath(p)
    rel = os.path.relpath(p, _REPO)
    return p if rel.startswith("..") else rel


def _tree_key_path(entry):
    """The list of string key names one ``_METADATA`` tree entry describes, else ``None``.

    Read from the STRUCTURED ``key_metadata`` list, never by parsing the entry's repr-shaped dict
    key (``"('params', 'vector_action_embedding', 'vec_act_proj', 'kernel')"``), which is a
    display form. ``None`` for any entry that is not a plain string-keyed path — orbax also
    records sequence/int keys, and nothing is inferred from those."""
    if not isinstance(entry, dict):
        return None
    key_metadata = entry.get("key_metadata")
    if not isinstance(key_metadata, list):
        return None
    path = []
    for part in key_metadata:
        if not isinstance(part, dict) or not isinstance(part.get("key"), str):
            return None
        path.append(part["key"])
    return path


def infer_vector_action(leaf_dir):
    """``(value, reason)`` — ``True``/``False`` when the checkpoint's OWN structure on disk
    answers whether it was saved from a vector-action architecture, else ``(None, why_not)``.

    INFER, DO NOT GUESS (TASK_889 / D39 scope 4, fix pass). `vector_action` decides whether
    `vector_action_embedding.vec_act_proj.{kernel,bias}` exist in the param tree at all, so the
    checkpoint on disk already carries the answer — asserting it from a default is asserting
    something the file itself can be asked.

    IT ASKS THE STRUCTURE, NOT THE ARRAYS (fix pass 3, the BLOCKER this rewrite closes). The
    previous draft asked the same question with a target-less
    `flax_checkpoints.restore_checkpoint(target=None)`, which DESERIALIZES every array and
    therefore depends on the SAVED SHARDING. `runs/832_e5/ckpt_imprint_full` — one of this
    tool's own seven default targets — was saved on a GPU host and records
    `device_str: "cuda:0"` in its `_sharding`; a target-less restore on this CPU-only
    workstation cannot concretize that and dies with `ValueError: sharding passed to
    deserialization should be specified, concrete and an instance of jax.sharding.Sharding. Got
    None`. The architecture was never ambiguous — that leaf's `_METADATA` plainly lists
    `('params', 'vector_action_embedding', 'vec_act_proj', ...)` — so a refusal there refused
    something KNOWABLE, on a device accident, and (before the batch fix in `main`) aborted the
    whole run.

    `_METADATA` is orbax's own listing of the key paths the save wrote. Reading it is a JSON
    parse of a ~270KB sidecar: no array is deserialized, no device is consulted, no sharding is
    concretized, and no model is built. It answers the presence question EXACTLY — a
    target-shaped read could not (it cannot detect a leaf the target lacks: that is
    `loaded_param_shape_defect`'s documented blind spot, and it is why the original hardcoded
    guess was invisible).

    Opened through `_read_json_sidecar`, so the stat-before-open discipline covers it like every
    other sidecar this program reads off a derived path. `None` is returned for every shape of
    "this read cannot answer it" (no/unsafe/unparseable `_METADATA`, no `tree_metadata` mapping,
    a `vector_action_embedding` subtree without the projection leaves) — and the caller then
    REFUSES, demanding an explicit assertion. It never falls back to a belief, and never falls
    back to the array read either: a mechanism that fails on device provenance must not be the
    thing that decides whether an architecture is knowable."""
    meta_path = os.path.join(leaf_dir, ORBAX_TREE_METADATA)
    doc, reason = _read_json_sidecar(meta_path, "a checkpoint leaf's orbax tree listing")
    if doc is None:
        return None, (reason or
                      f"{_display_path(leaf_dir)} carries no {ORBAX_TREE_METADATA} tree listing, "
                      f"so the architecture cannot be read off the structure orbax recorded")
    tree = doc.get("tree_metadata")
    if not isinstance(tree, dict):
        return None, (f"{_display_path(meta_path)} carries no `tree_metadata` mapping (got "
                      f"{type(tree).__name__}), so no saved key path can be read from it")
    prefix = ("params", VECTOR_ACTION_SUBTREE)
    subtree_seen = False
    for entry in tree.values():
        path = _tree_key_path(entry)
        if path is None or tuple(path[:2]) != prefix:
            continue
        subtree_seen = True
        if len(path) > 2 and path[2] == VECTOR_ACTION_LEAF_PARENT:
            return True, (f"{VECTOR_ACTION_SUBTREE}.{VECTOR_ACTION_LEAF_PARENT} is PRESENT in "
                          f"the saved params ({ORBAX_TREE_METADATA} key path "
                          f"{'.'.join(path)}) — this checkpoint was written by a vector-action "
                          f"architecture")
    if subtree_seen:
        return None, (f"the saved params carry a {VECTOR_ACTION_SUBTREE} subtree with no "
                      f"{VECTOR_ACTION_LEAF_PARENT} inside it — neither architecture looks like "
                      f"this, so nothing is inferred from it")
    return False, (f"{VECTOR_ACTION_SUBTREE} is ABSENT from the saved params (no such key path "
                   f"in {ORBAX_TREE_METADATA}) — this checkpoint was written by a scalar-action "
                   f"architecture")


def resolve_vector_action(meta, leaf_dir, asserted, ckpt_display):
    """``(value, how)`` — the architecture this checkpoint's target must be built with, from the
    checkpoint itself wherever possible. RAISES rather than returning a guess.

    Order: the checkpoint's own RECORDED ``vector_action`` (save-time ground truth), then
    INFERENCE from the parameters on disk, then the operator's explicit ``--vector-action`` /
    ``--no-vector-action`` assertion. Every DISAGREEMENT between two available sources refuses
    and writes nothing — a record that contradicts its own payload, or an operator assertion
    that contradicts either, is exactly the "the assumed architecture is wrong" signal this tool
    exists to produce, and the whole point of D39 scope 4 is that it must never be resolved by
    picking a side quietly."""
    recorded = meta.get("vector_action")
    inferred, why = infer_vector_action(leaf_dir)

    if recorded is not None and inferred is not None and bool(recorded) != bool(inferred):
        raise RuntimeError(
            f"REFUSING to stamp {ckpt_display}: its trainer_metadata.json records "
            f"vector_action={recorded!r}, but its own parameters on disk say {inferred!r} "
            f"({why}).\n"
            f"  Nothing written. A checkpoint whose recorded architecture contradicts its own "
            f"payload cannot be fingerprinted honestly under either value — resolve which is "
            f"true before stamping.")
    resolved = recorded if recorded is not None else inferred
    if resolved is not None:
        source = ("its own recorded vector_action" if recorded is not None
                  else f"INFERRED from the checkpoint's own structure on disk ({why})")
        if asserted is not None and bool(asserted) != bool(resolved):
            raise RuntimeError(
                f"REFUSING to stamp {ckpt_display}: you asserted "
                f"--{'' if asserted else 'no-'}vector-action, but the checkpoint says "
                f"vector_action={resolved!r} ({source}).\n"
                f"  Nothing written: an operator assertion that contradicts the checkpoint is "
                f"not an override, it is a disagreement — and stamping under the wrong "
                f"architecture writes a WRONG fingerprint into the document TASK_886 certifies "
                f"against (TASK_889 / D39 scope 4). Drop the flag to use the checkpoint's own "
                f"answer.")
        return bool(resolved), source
    if asserted is not None:
        return bool(asserted), (f"ASSERTED on the command line (--"
                                f"{'' if asserted else 'no-'}vector-action); the checkpoint "
                                f"records nothing and {why}")
    raise RuntimeError(
        f"REFUSING to stamp {ckpt_display}: its architecture is UNKNOWN — trainer_metadata.json "
        f"records no `vector_action`, and it could not be inferred from the checkpoint's own "
        f"parameters ({why}).\n"
        f"  Nothing written, and nothing assumed: the previous draft applied a hardcoded default "
        f"here and SILENTLY accepted whatever fingerprint that produced, which is the defect "
        f"D39 scope 4 names. Re-run with an explicit --vector-action or --no-vector-action if "
        f"you can assert the architecture yourself.")


def _build_compatible_trainer(meta, n_assets, vector_action, seed=0):
    """Reconstruct a trainer whose architecture matches ``meta`` (a checkpoint's OWN
    trainer_metadata.json) exactly enough for checkpoint_io.load_checkpoint_into's shape check
    to pass. ``model_size="small"`` is a neutral placeholder: TrainingConfig only remaps
    state_dim from model_size when state_dim is left at ITS OWN default AND model_size != "small"
    (config.py __post_init__), so passing model_size="small" alongside an explicit state_dim
    means the explicit value always wins, whatever it is.

    ``vector_action`` (TASK_889 / D39 B2) is a REQUIRED caller-supplied argument, never a
    hardcode inside this function — see ``resolve_vector_action`` for where it comes from (the
    checkpoint's own recorded value, else INFERRED from its parameters on disk, else an explicit
    operator assertion; unknowable is REFUSED, never defaulted)."""
    cfg = TrainingConfig(
        name="889_stamp",
        model_size="small",
        state_dim=int(meta["state_dim"]),
        h_cycles=int(meta["h_cycles"]),
        l_cycles=int(meta["l_cycles"]),
        use_prediction_loss=bool(meta.get("use_prediction_head", False)),
        physics_only=bool(meta.get("physics_only", False)),
        vector_action=bool(vector_action),
        seed=seed,
    )
    model, trainer = setup_model_and_trainer(cfg, n_assets)
    dummy_obs = jnp.ones((1, 27, n_assets, n_assets))
    trainer.init_train_state(jax.random.PRNGKey(seed), dummy_obs)
    return trainer


def _atomic_write_json(path, doc):
    """Read-modify-write with an atomic swap (tmp + os.replace) — a crash mid-write leaves the
    ORIGINAL file intact, never a truncated one. Same idiom this codebase already uses for
    other provenance sidecars (imprint.py's frame_provenance writer)."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp_path, path)


def _digest_pins_naming(path):
    """The hand-kept ``*.sha256`` pins a write to ``path`` would INVALIDATE, for display.

    The rule itself lives in ``checkpoint_io.digest_pins_naming`` — ONE definition shared with
    ``imprint.py::write_frame_provenance``, the other writer that can move a pinned provenance
    document (a second copy of "which digest covers this file" drifts from the first exactly the
    way the guard drifted from the loader). This wrapper only relativizes for display."""
    return [_display_path(p) for p in checkpoint_io.digest_pins_naming(path)]


def _provenance_names_run_dir(doc, run_dir):
    """True iff ``doc``'s own ``"checkpoint"`` field RESOLVES to ``run_dir``.

    ``imprint.build_frame_provenance`` writes that field as a repo-relative display path
    (``runs/832_e5/ckpt_imprint_full``), so it is resolved here through
    ``checkpoint_io.resolve_checkpoint_dirs`` — the ONE resolution function — which accepts
    either committed spelling (run dir or orbax leaf) and normalises symlinks the same way the
    loader does. A document with no ``"checkpoint"`` field (or a non-string one) names nothing
    and therefore binds to nothing: False."""
    recorded = doc.get("checkpoint")
    if not isinstance(recorded, str) or not recorded:
        return False
    abs_recorded = recorded if os.path.isabs(recorded) else os.path.join(_REPO, recorded)
    recorded_run, _leaf = checkpoint_io.resolve_checkpoint_dirs(abs_recorded)
    return recorded_run == run_dir


def _locate_provenance(run_dir):
    """``(path, note)`` — the frame-provenance document that describes ``run_dir``.

    ``checkpoint_io.provenance_path_for`` derives the name from the CHECKPOINT's basename, and
    that derivation MISSES the single most-cited production checkpoint:
    ``runs/832_e5/ckpt_imprint_full`` derives ``frame_provenance_imprint_full.json``, which does
    not exist — the real document is ``runs/832_e5/frame_provenance_imprint.json``, and its own
    ``"checkpoint"`` field says ``runs/832_e5/ckpt_imprint_full``.

    The repair lives HERE, in the one-shot stamp, and deliberately NOT in
    ``provenance_path_for``: that function is also what ``eval_vector.py`` uses to find
    provenance for a GRADING run, and changing it would change which provenance a graded run
    finds — out of this row's scope.

    Rule: the derived name wins whenever it exists (every future checkpoint, and the three
    ``runs/861_ref`` survivors). Only when it is ABSENT does this fall back to reading the
    sibling ``frame_provenance*.json`` documents and asking each one which checkpoint IT says it
    describes. TWO candidates naming the same run dir is a REFUSAL, not a pick: a repair tool
    that guesses which of two documents describes these weights is the fail-open this task
    exists to close (and it aborts the whole run — ``preflight_checkpoint_loadable``'s
    documented conservative direction: a batch that silently drops or mis-binds a member
    produces an unstated result)."""
    derived = checkpoint_io.provenance_path_for(run_dir)
    if os.path.lexists(derived):
        return derived, None
    parent = os.path.dirname(run_dir) or "."
    try:
        names = sorted(os.listdir(parent))
    except OSError:
        return derived, None
    hits = []
    for name in names:
        if not (name.startswith("frame_provenance") and name.endswith(".json")):
            continue
        candidate = os.path.join(parent, name)
        doc, _reason = _read_json_sidecar(candidate, "a checkpoint's frame provenance")
        if doc is not None and _provenance_names_run_dir(doc, run_dir):
            hits.append(candidate)
    if len(hits) > 1:
        raise RuntimeError(
            f"AMBIGUOUS frame provenance for {_display_path(run_dir)}: "
            f"{len(hits)} sibling documents each record this checkpoint — "
            f"{', '.join(_display_path(h) for h in hits)}.\n"
            f"  Refusing to pick one: binding a fingerprint into the wrong document would make "
            f"TASK_886's contract certify these weights against a frame they were not trained "
            f"on. Resolve the duplication by hand, then re-run.")
    if len(hits) == 1:
        return hits[0], (
            f"the derived name {os.path.basename(derived)} does not exist; using "
            f"{_display_path(hits[0])}, whose own \"checkpoint\" field names this run dir "
            f"(provenance_path_for's documented non-injective naming rule does not reach this "
            f"checkpoint — see _locate_provenance)")
    return derived, None


def _read_json_sidecar(path, what):
    """``(doc, reason)`` — exactly one is None. Absence is ``(None, None)``.

    ``stat`` before ``open``, through the SAME ``checkpoint_io`` helper every other sidecar read
    in this program uses. These files are RUN-DIR SIBLINGS of the orbax leaf, so the load
    guard's leaf walk never covered them; a FIFO here blocked the reader forever."""
    if not os.path.lexists(path):
        return None, None
    defect = checkpoint_io.unsafe_to_open_defect(
        path, checkpoint_io.SIDECAR_MAX_BYTES, what)
    if defect is not None:
        return None, f"{os.path.basename(path)} is not safe to open — {defect}"
    try:
        with open(path) as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as e:
        return None, f"{os.path.basename(path)} unreadable ({type(e).__name__}: {e})"
    if not isinstance(doc, dict):
        return None, (f"{os.path.basename(path)} is not a JSON object "
                      f"(got {type(doc).__name__})")
    return doc, None


def stamp_one(ckpt_arg, n_assets, vector_action_assert=None, apply=False):
    """Returns a result dict; never raises for a MISSING checkpoint or missing/unreadable
    metadata (both are reported, not fatal — "a checkpoint you do not have cannot be graded").
    RAISES for: a checkpoint that IS present, IS resolvable, and STILL fails to load or to write
    its sidecar (a genuine defect worth stopping the run for); a provenance document that names
    a DIFFERENT checkpoint (refusing to re-bind it); or — TASK_889 / D39 B2 — a RECORDED
    ``param_fingerprint`` that DISAGREES with the freshly recomputed one, in either sidecar. That
    last case writes NOTHING, in either file, whether or not ``--apply`` was passed: a
    disagreement is a defect worth stopping the whole run for, not something that should need
    ``--apply`` to surface.

    ``apply=False`` (the default) computes and REPORTS everything and writes nothing."""
    run_dir, leaf_dir = checkpoint_io.resolve_checkpoint_dirs(ckpt_arg)
    if leaf_dir is None:
        # Clipped: `run_dir` is derived from an operator-supplied `--checkpoint`, which is
        # bounded only by MAX_ARG_STRLEN — the same unbounded echo checkpoint_io refuses to
        # print. The path stays identifiable; only how much of it is spelled is capped.
        return {"checkpoint": ckpt_arg, "status": "skipped",
                "reason": f"no loadable checkpoint at resolved path "
                          f"{checkpoint_io.clip_for_refusal(run_dir)}"}
    meta_path = os.path.join(run_dir, "trainer_metadata.json")
    meta, meta_reason = _read_json_sidecar(meta_path, "a checkpoint's trainer_metadata.json")
    if meta is None:
        return {"checkpoint": ckpt_arg, "status": "skipped",
                "reason": meta_reason or
                          (f"no trainer_metadata.json beside {_display_path(run_dir)} — cannot "
                           f"reconstruct a compatible architecture without guessing")}
    required = ("h_cycles", "l_cycles", "state_dim")
    missing = [k for k in required if meta.get(k) is None]
    if missing:
        return {"checkpoint": ckpt_arg, "status": "skipped",
                "reason": f"trainer_metadata.json is missing {missing} — cannot reconstruct "
                          f"architecture without guessing"}

    # TASK_889 / D39 B2 (fix pass): NEVER a hardcode and never a silent default — RECORDED,
    # else INFERRED from the parameters on disk, else the operator's explicit assertion, else
    # REFUSED. Every disagreement between those sources refuses and writes nothing. This is the
    # one decision that determines the TARGET the fingerprint is computed over, so getting it
    # wrong writes a well-formed WRONG value into the document TASK_886 certifies against.
    vector_action, vector_action_source = resolve_vector_action(
        meta, leaf_dir, vector_action_assert, _display_path(run_dir))

    # The provenance document is located BEFORE the (expensive) load, so a dry run reports the
    # full picture — including "this checkpoint has no provenance and therefore stays
    # uncertifiable" — without paying for weights it is not going to record anywhere.
    prov_path, prov_note = _locate_provenance(run_dir)
    prov, prov_reason = _read_json_sidecar(prov_path, "a checkpoint's frame provenance")
    if prov is None and prov_reason is None:
        prov_reason = (f"no {os.path.basename(prov_path)} beside "
                       f"{_display_path(run_dir)} — this checkpoint stays UNCERTIFIABLE under "
                       f"TASK_886's contract (it reads the fingerprint from the provenance "
                       f"document); nothing is synthesized here")
    elif prov is not None and prov_note:
        prov_reason = prov_note

    trainer = _build_compatible_trainer(meta, n_assets, vector_action)
    # RAISES on a genuine defect (foreign-shape restore / no-op alias / hang-risk leaf) —
    # not caught here: those are real problems the operator must see, not a skip.
    #
    # allow_architecture_mismatch=True (TASK_889): this tool's OWN job is exactly "compute a
    # fingerprint under an assumed architecture and compare it against what is on record" — the
    # disagreement check a few lines below IS that comparison, with a tool-specific message and
    # a "write nothing" contract. Without this flag, trainer.load_checkpoint's own hard refusal
    # (added by this same task) would fire on the IDENTICAL comparison first — same metadata
    # field, same recomputed fingerprint — and this tool's own check could never be reached. The
    # vector_action/use_prediction_head checks that flag also gates never fire for a trainer
    # THIS tool built: `vector_action` above and `_build_compatible_trainer`'s
    # `use_prediction_loss` both already copy whatever IS recorded, so there is never a
    # target-vs-recorded disagreement of that kind to suppress.
    evidence = checkpoint_io.load_checkpoint_into(trainer, ckpt_arg,
                                                  allow_architecture_mismatch=True)
    fp = evidence["param_fingerprint"]
    field = checkpoint_io.PROVENANCE_FINGERPRINT_FIELD

    # The document must DESCRIBE the weights just loaded, or nothing is merged into it.
    # `provenance_path_for` is documented non-injective (`.../ckpt_foo` and `.../foo` land on
    # the same file) and `imprint.write_frame_provenance` defends that with a differing-document
    # refusal — this read-merge-write needs the same defence.
    if prov is not None and not _provenance_names_run_dir(prov, run_dir):
        raise RuntimeError(
            f"REFUSING to stamp a fingerprint into {_display_path(prov_path)}: that document's "
            f"own \"checkpoint\" field is {prov.get('checkpoint')!r}, which does not resolve to "
            f"the checkpoint just loaded ({_display_path(run_dir)}, leaf "
            f"{_display_path(evidence['resolved_path'])}).\n"
            f"  provenance_path_for's naming rule is NOT injective, so two checkpoints under "
            f"one parent can derive the SAME document — writing here would re-bind another "
            f"checkpoint's provenance to these weights. Nothing was written.")

    # TASK_889 / D39 B2: a RECORDED fingerprint that DISAGREES with the RECOMPUTED one is the
    # exact "a checkpoint changed underneath you — or the assumed architecture was wrong" signal
    # this tool exists to produce. REFUSE and write NOTHING for this checkpoint (neither
    # sidecar) — never silently overwrite it, in dry run OR --apply.
    recorded_meta_fp = meta.get(field)
    if recorded_meta_fp is not None and recorded_meta_fp != fp:
        raise RuntimeError(
            f"REFUSING to stamp {_display_path(meta_path)}: its recorded {field}="
            f"{recorded_meta_fp!r} disagrees with the recomputed {fp!r} (target built with "
            f"vector_action={vector_action!r}, h_cycles={meta['h_cycles']}, "
            f"l_cycles={meta['l_cycles']}, state_dim={meta['state_dim']}).\n"
            f"  Nothing written, in either sidecar: either the assumed architecture is wrong, "
            f"or this checkpoint's payload changed underneath its own recorded evidence — both "
            f"are worth stopping the run for, never worth silently overwriting.")
    if prov is not None:
        recorded_prov_fp = prov.get(field)
        if recorded_prov_fp is not None and recorded_prov_fp != fp:
            raise RuntimeError(
                f"REFUSING to stamp {_display_path(prov_path)}: its recorded {field}="
                f"{recorded_prov_fp!r} disagrees with the recomputed {fp!r}.\n"
                f"  Nothing written, in either sidecar — see the trainer_metadata.json check "
                f"above for what a disagreement here means.")

    writes = []
    if meta.get(field) != fp or meta.get("vector_action") != vector_action:
        writes.append((meta_path, {**meta, field: fp, "vector_action": vector_action},
                       "trainer_metadata.json"))
    if prov is not None and prov.get(field) != fp:
        writes.append((prov_path, {**prov, field: fp}, os.path.basename(prov_path)))
    # Digest pins are computed for the files that are ACTUALLY going to change.
    invalidated = sorted({d for path, _doc, _label in writes
                          for d in _digest_pins_naming(path)})
    if apply:
        for path, doc, _label in writes:
            _atomic_write_json(path, doc)
    return {"checkpoint": ckpt_arg, "status": "stamped" if apply else "would-stamp",
            "resolved_path": _display_path(evidence["resolved_path"]),
            "step": evidence["step"], "param_fingerprint": fp, "vector_action": vector_action,
            "vector_action_source": vector_action_source,
            "wrote": [label for _p, _d, label in writes],
            "provenance": _display_path(prov_path) if prov is not None else None,
            "provenance_note": prov_reason,
            "invalidated_digest_pins": invalidated,
            "unchanged": not writes}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", action="append", default=[],
                    help="checkpoint dir to stamp (repeatable). Default: the 7 checkpoints "
                         "named in TASK_888/889.")
    ap.add_argument("--n-assets", type=int, default=DEFAULT_N_ASSETS,
                    help=f"universe size to reconstruct the model with (default "
                         f"{DEFAULT_N_ASSETS} — the real-battery convention; NOT recorded in "
                         f"trainer_metadata.json, so this cannot be derived, only asserted)")
    ap.add_argument("--vector-action", dest="vector_action", action="store_true", default=None,
                    help="LAST-RESORT architecture assertion, used ONLY when a checkpoint "
                         "neither records `vector_action` nor allows it to be inferred from its "
                         "own parameters on disk. There is NO default (TASK_889 / D39 scope 4): "
                         "an unknowable architecture is REFUSED, never guessed, and an "
                         "assertion that contradicts the checkpoint is refused too.")
    ap.add_argument("--no-vector-action", dest="vector_action", action="store_false",
                    help="the same last-resort assertion, for a scalar-architecture checkpoint.")
    ap.add_argument("--apply", action="store_true", default=False,
                    help="actually WRITE the stamps. Default OFF = dry run: the targets are "
                         "the 7 irreplaceable live checkpoints, so a bare invocation must not "
                         "mutate them (this repo's --write-parity-baseline / "
                         "--promote-baselines convention). PLANNER-GATED against runs/.")
    args = ap.parse_args()

    targets = [os.path.join(_REPO, p) for p in DEFAULT_CHECKPOINTS] if not args.checkpoint \
        else args.checkpoint

    print(f"[stamp] {'APPLY — sidecars WILL be written' if args.apply else 'DRY RUN — nothing will be written (pass --apply to write)'}")
    results = []
    for ckpt in targets:
        # Clipping: `ckpt` is raw argv on the --checkpoint path (bounded only by
        # MAX_ARG_STRLEN); printing it unclipped would defeat checkpoint_io's own refusal-echo
        # discipline.
        print(f"[stamp] {checkpoint_io.clip_for_refusal(ckpt, quote=False)} ...")
        try:
            result = stamp_one(ckpt, args.n_assets, args.vector_action, apply=args.apply)
        except Exception as e:
            # ONE MEMBER'S REFUSAL DOES NOT ABORT THE BATCH (TASK_889 fix pass 3). Every refusal
            # `stamp_one` raises is deliberate and fail-closed and stays that way — nothing is
            # written for this checkpoint — but the DEFAULT invocation is a 7-member batch over
            # irreplaceable checkpoints, and dying on member 7 threw away the six verdicts
            # already computed and told the operator nothing about members that were never
            # reached. Classify, print in full, keep going; the exit code below carries the
            # failure so no script can read "finished" as "all clean". `Exception` deliberately
            # does not swallow KeyboardInterrupt/SystemExit.
            #
            # `RuntimeError` is this tool's OWN refusal vocabulary (every `raise` in `stamp_one`
            # and `resolve_vector_action` is one). Anything else is an UNEXPECTED failure, and
            # for those the traceback is kept and printed — classifying a genuine crash as a
            # tidy one-line "REFUSED" would trade one information loss (six discarded verdicts)
            # for another (a stack that no longer reaches stderr).
            result = {"checkpoint": ckpt, "status": "refused",
                      "reason": f"{type(e).__name__}: "
                                f"{checkpoint_io.clip_for_refusal(e, limit=REFUSAL_BODY_MAX, quote=False)}",
                      "traceback": None if isinstance(e, RuntimeError) else traceback.format_exc()}
        results.append(result)
        if result["status"] == "skipped":
            print(f"  SKIPPED: {result['reason']}")
            continue
        if result["status"] == "refused":
            print(f"  REFUSED: {result['reason']}")
            if result.get("traceback"):
                print("  ^^ UNEXPECTED — not one of this tool's own refusals. Full traceback:")
                print(result["traceback"])
            continue
        verb = "stamped" if args.apply else "WOULD stamp"
        note = " (already current — idempotent re-run, nothing to write)" \
            if result.get("unchanged") else f" -> {', '.join(result['wrote'])}"
        print(f"  {verb}: resolved={result['resolved_path']} step={result['step']} "
              f"vector_action={result['vector_action']} "
              f"param_fingerprint={result['param_fingerprint'][:16]}...{note}")
        print(f"  vector_action came from: {result['vector_action_source']}")
        if result.get("provenance_note"):
            print(f"  NOTE: {result['provenance_note']}")
        for digest in result.get("invalidated_digest_pins") or []:
            print(f"  {'!!! INVALIDATED' if args.apply else '!!! WOULD INVALIDATE'} the "
                  f"hand-kept digest pin {digest} — this repair changes the bytes it records. "
                  f"This tool does NOT rewrite it (a digest a repair tool refreshes on its own "
                  f"behalf pins nothing); re-pin it deliberately.")

    n_skipped = sum(1 for r in results if r["status"] == "skipped")
    n_refused = sum(1 for r in results if r["status"] == "refused")
    n_done = len(results) - n_skipped - n_refused
    n_prov = sum(1 for r in results if r.get("provenance"))
    print("=" * 84)
    print(f"[stamp] {n_done} {'stamped' if args.apply else 'would be stamped'}, "
          f"{n_skipped} skipped (missing/unreconstructable — not a failure), "
          f"{n_refused} REFUSED (nothing written for those — each reason is printed above), "
          f"{len(results)} total; {n_prov}/{n_done} have a frame-provenance document (only "
          f"those are certifiable under TASK_886's contract).")
    if not args.apply:
        print("[stamp] DRY RUN — nothing was written. Re-run with --apply to write.")
    if n_refused:
        print(f"[stamp] EXIT 1 — {n_refused} of {len(results)} checkpoint(s) REFUSED. The other "
              f"members' results above are complete and stand on their own; a refusal is a "
              f"defect to resolve, never a member to skip past silently.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
