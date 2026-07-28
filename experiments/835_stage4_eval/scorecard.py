#!/usr/bin/env python3
"""TASK_862 / imprint benchmark scorecard — the full battery as ONE honest artifact.

The imprint (``runs/832_e5/ckpt_imprint_full``) is the validated frontrunner. Per the D17
correction it owes the FULL benchmark battery BEFORE the 4-gate ladder: the four DEC-2 regime
windows (each against its own per-window floor artifact in ``runs/831_ext/``), the full OOS
window, and the pre-registered walk-forward quarters
(``runs/858_oos/WALKFORWARD_PREREG.md``). Until now each of those was a hand-assembled
``eval_vector.py`` invocation with its numbers scattered across ``runs/`` JSONs. This module
codifies the battery: ONE command, ONE scorecard artifact, a STABLE schema the eventual
gate-ladder submission can cite.

WHAT IT DOES NOT DO: it does not re-implement (or edit) any evaluation machinery. Every window
is graded by the SHIPPED ``experiments/835_stage4_eval/eval_vector.py``, invoked as a
SUBPROCESS in the 848 read-of-record configuration and read back from its own emitted JSON.
This harness is a driver + a schema, nothing else. (TASK_861 edits ``eval_vector.py`` /
``imprint.py`` concurrently; this row stays merge-safe by adding files only — hence
subprocess-invocation instead of an import.)

-------------------------------------------------------------------------------------------
OPERATOR: PINNED IN ARGV, THEN **OBSERVED** BACK OUT OF THE RESULT
-------------------------------------------------------------------------------------------
The operator is the 848 read of record (LEDGER 832/E5-oos-real, ``runs/858_oos/
manifest_oos.txt``) and is FIXED in code — deliberately not a manifest knob::

    eval_vector.py --db <db> --start <s> --end <e> --bar-minutes 1440
                   --n-assets <n> --eval-seed <seed> --checkpoint <ckpt>
                   --panel-frame-from-checkpoint <frame>
                   --operator policy-argmax --teacher-floor --vector-weight-rebalance
                   --decision-rule argmax
                   --position-size-pct 0.1 --max-positions 16 --min-cash-threshold 100.0

which yields BOTH cost tiers (fee0 gross and fee21 = fee16+slip5 net) per window in one run.

An operator field is only trustworthy if the artifact can tell it apart from an argparse
default that moved under us. Two mechanisms, and NEITHER is an echo of our own argv:

  * **Pinned in argv** — the fields eval_vector does NOT report back in its result JSON
    (``max_positions``, ``min_cash_threshold``) plus the ones whose default drifting would
    silently change the read (``decision_rule``, ``position_size_pct``). Passing them
    explicitly at their 848 values keeps the read stable even if a default moves.
  * **Observed + asserted** — every field eval_vector DOES record in ``result["operator"]``
    is copied VERBATIM into the window row and compared field-by-field against
    ``OPERATOR_PIN`` below; ``result["panel"]`` is compared against ``PANEL_PIN``
    (position_size_pct, warm) and ``result["checkpoint"]`` against the manifest. Any
    disagreement is a FAILED row, and because a moved default poisons every window equally
    the battery ABORTS the remaining windows rather than burning hours on them.
    Operator fields eval_vector reports that the pin does not know about (e.g. a knob a later
    task adds) are NOT fatal: they are listed per row under ``operator_unpinned`` and marked
    in the MD, so a new knob is visible instead of invisible.

-------------------------------------------------------------------------------------------
FAIL-LOUD CONTRACT (no partial-success masquerade, and no lost battery)
-------------------------------------------------------------------------------------------
  * Global preflight — missing db / missing checkpoint dir / unreadable frame provenance /
    a frame that mismatches the run (length != n_assets, duplicate slots, a recorded
    ``checkpoint`` or ``n_assets`` that disagrees with the manifest) — REFUSES: no window is
    run, EVERY window row is emitted as ``FAILED`` with that reason, exit code 1.
  * Per window — a missing/unreadable/mis-windowed/non-numeric floor artifact, a regime
    window with no floor artifact declared, an ``eval_vector`` non-zero exit or TIMEOUT, a
    missing result JSON, or a result whose provenance disagrees with the manifest (db /
    window / operator pin / checkpoint / frame / n_assets / panel pin / net-dependence /
    cost tiers) — is a ``FAILED`` row in the artifact.
  * ANY other exception raised while grading a window (OSError, a schema drift in
    eval_vector's own JSON, anything) is ALSO caught and becomes a FAILED row carrying the
    exception type + a traceback tail. A window is NEVER silently skipped, and a late failure
    NEVER discards the windows that already completed.
  * The artifact is ALWAYS written — including when the writer's own schema self-check fails
    (the offending row is degraded to FAILED; the header records the self-check errors).
  * ANY failed row, global error, or self-check failure makes the process exit non-zero.

NOT a guard, despite appearances: the floor cross-check in ``run_window`` (pinned artifact vs
eval_vector's own 831-EXT per-window floor) is TAUTOLOGICAL in the canonical configuration —
both sides read the same ``runs/831_ext/<name>.json``. It bites only when a manifest pins a
DIFFERENT floor file than the one eval_vector's own 831-EXT table resolves for that window
name. Do not read it as independent corroboration of a floor value.

DETERMINISM: the artifact carries no wall-clock stamp; two runs on the same inputs (same
manifest, same repo state, same CLI) are BYTE-IDENTICAL. Paths are repo-relative where they
live inside the repo; out-of-repo inputs/outputs (a manifest, floor artifact or out-dir under
/tmp, a db outside the tree) are necessarily recorded as absolute paths, so an artifact
produced from out-of-repo inputs is byte-stable only for a fixed choice of those paths.

CHILD ENV (recorded verbatim in the header as ``child_env``): per-window subprocesses are
FORCED — not defaulted — to ``XLA_PYTHON_CLIENT_MEM_FRACTION=0.7`` (project invariant 5) and to
``JAX_PLATFORMS=`` the value of ``--jax-platforms``, or ``"cpu"`` when the flag is absent.
Both are forced because the caller's shell could otherwise silently change the run: an ambient
*empty* ``JAX_PLATFORMS`` satisfies ``eval_vector``'s own ``os.environ.setdefault`` and moved a
whole battery onto the GPU while this header recorded ``jax_platforms: null`` (different
floats, unrecorded, and a bare GPU launch). Caveats that remain: ``XLA_PYTHON_CLIENT_PREALLOCATE``
is still inherited from the caller and is not part of ``child_env``; and because ``eval_vector``
sets ``PREALLOCATE=false``, ``MEM_FRACTION`` sizes an absent pool rather than enforcing a cap —
it is NOT a substitute for a cgroup. The harness imposes no RAM cap: launch the battery itself
under ``safe_run`` (the ``runs/858_oos/manifest_oos.txt`` convention: ``safe_run gpu 16G``) for
any GPU run.

-------------------------------------------------------------------------------------------
MANIFEST SCHEMA  ``imprint_scorecard_manifest/v1``  (the canonical instance ships alongside:
``experiments/835_stage4_eval/scorecard_manifest_imprint.json``)
-------------------------------------------------------------------------------------------
Unknown keys are REJECTED (a typo'd ``"eval_seeds"`` must not quietly grade at seed 0).
Object keys (paths are repo-relative unless absolute)::

  schema            str   REQUIRED, == "imprint_scorecard_manifest/v1"
  name              str   REQUIRED, artifact basename (scorecard_<name>.json/.md); [A-Za-z0-9_.-]
  checkpoint        str   REQUIRED, checkpoint directory to grade
  frame_provenance  str   REQUIRED, JSON recording the checkpoint's ORDERED training frame
                          ('frame_symbols' or 'symbols') — TASK_853 fixed-frame panel
  db                str   REQUIRED, sqlite panel source
  n_assets          int   REQUIRED, must equal the recorded frame length
  eval_seed         int   optional, default 0
  model             obj   optional {model_size:str, h_cycles:int, l_cycles:int} — only passed
                          through when present (else eval_vector's defaults: small / 4 / 6).
                          model_size is validated against ('small','medium','large'): the
                          trainer maps anything else silently onto the small architecture.
  notes             str   optional, carried into the artifact header verbatim
  windows           list  REQUIRED, >=1 window objects:
      name            str  REQUIRED, unique, [A-Za-z0-9_.-], not a path component like '..'
      kind            str  REQUIRED, one of "regime" | "oos" | "walkforward"
      start,end       str  REQUIRED, YYYY-MM-DD (start < end)
      floor_artifact  str  REQUIRED for kind=="regime" (a runs/831_ext/<name>.json carrying
                           {"window":[s,e], "floor":{"0":..,"21":..}, "ceiling":{...}});
                           optional otherwise
      max_decisions   int  optional, cap decisions (fixture smoke only; omit for real reads).
                           A capped window is marked CAPPED in both renders — a stray cap must
                           never read as a full window.

-------------------------------------------------------------------------------------------
SCORECARD SCHEMA  ``imprint_scorecard/v1``  (``scorecard_<name>.json``; the .md is a render)
-------------------------------------------------------------------------------------------
Header::

  schema, generated_by            str
  manifest      {path, name, schema, sha256}
  checkpoint    str
  frame_provenance {path, n_frame_symbols, meta}   meta = every non-frame key of the
                                                    provenance JSON (train_window, db, source…)
  db, n_assets, bar_minutes, eval_seed, model, jax_platforms, notes
  child_env     {VAR: val}           env pinned onto every eval_vector subprocess
  window_timeout_s  int
  capped_windows    list[str]        windows run under a max_decisions cap ([] on a real read)
  operator      {mode, teacher_floor, vector_weight_rebalance, panel_frame_from_checkpoint,
                 decision_rule, cost_tiers_bp, argv_template, source, pin, panel_pin,
                 verification}       pin/panel_pin = the asserted 848 field set (see above)
  git_describe  str|null
  status        "OK" | "FAILED"      ("FAILED" iff any row failed / global error / self-check)
  n_windows, n_ok, n_failed          int
  global_errors list[str]            non-empty only on a refused (preflight) run
  self_check    {ok, errors, degraded_rows}
  cross_config_floor  bool           true iff any pinned floor was measured under a different
                                     configuration than this battery (see row config_delta)
  consistency   {form, dec2_applicable, floor_bases, n_regime_windows, n_regime_ok,
                 n_at_or_above_regime_floor, n_at_or_above_teacher_floor, required,
                 primary_basis, meets_form_teacher_floor,
                 meets_form_pinned_floor_cross_config, d17, note}
                 meets_form* are null unless every regime window ran
  windows       list[row]

Window row (status "OK")::

  name, kind, status, window[start,end], n_decisions, steps_completed, partial, max_decisions
  panel   {n_rows, n_assets, n_active_slots, n_inactive_slots, n_partial_slots,
           position_size_pct, warm}
  operator          dict   eval_vector's own result["operator"], VERBATIM (observed, not echoed)
  operator_unpinned list   operator fields present in the result but absent from OPERATOR_PIN
  tiers   {"0": tier, "21": tier}   tier = {
              sharpe, teacher_floor, delta_vs_teacher_floor, final_pv, bankrupt_at, bankrupt,
              terminated_early_at, pct_hold_agg,
              activity {hold_pct, trades_per_step, n_trades_total, n_long, n_short,
                        active_assets, top5_traded}      # the TASK_859 block, verbatim
          }
  regime_floor  null, or {path, sha256, floor{tier:val}, ceiling{tier:val}, source_meta,
                          config_delta{differences[str], cross_config bool, fields{...}},
                          delta_vs_regime_floor{tier:val}, at_or_above_regime_floor_net21}
  hold_collapse, net_dependent   bool
  eval_artifact, eval_log        str   (paths to eval_vector's own JSON + captured console)

Window row (status "FAILED") carries: name, kind, status, window, reason (str), plus
``eval_log`` when a subprocess ran. Scored keys are ABSENT — a failed window has no numbers.

READING THE FLOOR COLUMNS — TWO BASES, ONLY ONE OF THEM APPLES-TO-APPLES:

  * ``meets_form_teacher_floor`` (**the sound basis**) uses the ON-WINDOW teacher floor
    computed by the SAME run: same universe, same panel, same operator, same costs.
  * ``meets_form_pinned_floor_cross_config`` uses the manifest's pinned ``runs/831_ext``
    artifacts. Those floors were measured under a materially DIFFERENT configuration
    (hundreds of assets vs this battery's n_assets, position_size_pct ~0.002-0.005 vs 0.1,
    warm 31 vs 60, the server db, a dithered-book search grid over many seeds). The breadth
    delta alone is first-order. Every such difference is enumerated per row under
    ``regime_floor.config_delta`` and rendered as a CONFIG-DELTA block in the MD; the header
    flag ``cross_config_floor`` says whether any window is affected. Reconciling the
    comparison is a planner decision, NOT something this harness may paper over.

``validate_scorecard(obj)`` in this module is the schema checker (raises on violation); the
writer self-checks with it before emitting (degrading, never suppressing, on violation) and
the fixture smoke asserts on it.

FACTORY / SCHEDULED split (830/831 house style): this module + its fixture smoke
(``scorecard_smoke.py``) are the factory deliverable. The REAL battery
(``scorecard_manifest_imprint.json`` against the read-only ``trading_data.db``) is a SCHEDULED
planner run — never inside the factory loop. Per D17 a consistency PASS confers NO gate
status; it completes the battery prerequisite.
"""
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import traceback
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
EVAL_VECTOR = os.path.join(_HERE, "eval_vector.py")

MANIFEST_SCHEMA = "imprint_scorecard_manifest/v1"
SCORECARD_SCHEMA = "imprint_scorecard/v1"
TIERS = ("0", "21")                      # eval_vector.GATE_COSTS as JSON keys: gross, net@21
NET_TIER = "21"
BAR_MINUTES = 1440                       # the fixed-frame daily panel rule (TASK_853)
WINDOW_KINDS = ("regime", "oos", "walkforward")
DEC2_CONSISTENCY_MIN = 3                 # ">=3 of 4 regime windows at/above their window floor"
DEC2_N_REGIME = 4                        # the DEC-2 form is defined over exactly 4 windows
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
MODEL_SIZES = ("small", "medium", "large")   # mcts_trader/CLAUDE.md; anything else maps to small
DEFAULT_WINDOW_TIMEOUT_S = 6 * 3600      # generous; a hung eval must not block a scheduled run
XLA_MEM_FRACTION = "0.7"                 # project invariant 5 — no bare GPU child
# Pinned explicitly so an ambient JAX_PLATFORMS can never silently move the battery onto another
# backend while the header records "null". CPU is the honest default for this harness: the real
# battery is 7 sequential evals and `--jax-platforms cuda` must be an explicit, recorded choice
# (and, per invariant 5, launched under safe_run — MEM_FRACTION alone is not a cap because
# eval_vector sets XLA_PYTHON_CLIENT_PREALLOCATE=false).
DEFAULT_JAX_PLATFORMS = "cpu"

MANIFEST_KEYS = ("schema", "name", "checkpoint", "frame_provenance", "db", "n_assets",
                 "eval_seed", "model", "notes", "windows")
WINDOW_KEYS = ("name", "kind", "start", "end", "floor_artifact", "max_decisions")

# The 848 read-of-record operator — FIXED here, deliberately NOT a manifest knob.
# argv pin: what eval_vector does not report back, plus the defaults whose drift would
# silently change the read.
OPERATOR_ARGS = ["--operator", "policy-argmax",
                 "--teacher-floor", "--vector-weight-rebalance",
                 "--decision-rule", "argmax",
                 "--position-size-pct", "0.1",
                 "--max-positions", "16",
                 "--min-cash-threshold", "100.0"]
OPERATOR_SOURCE = ("848 read of record — LEDGER 832/E5-oos-real; "
                   "runs/858_oos/manifest_oos.txt")

# Observed pin: every field eval_vector records in result["operator"], with its 848 value.
# A window whose result disagrees with ANY of these is FAILED (and aborts the battery).
OPERATOR_PIN = {
    "mode": "policy-argmax",
    "vector_k": 16,
    "sims": 32,
    "hold_bias": 3.0,
    "dither_scale": 1.0,
    "proposal_scheme": "dithered-book",
    "proposal_temp": 1.0,
    "proposal_explore_slots": 2,
    "prior_weight": 1.0,
    "per_node_resampling": False,
    "vector_budget_step": False,
    "vector_budget_pct": 0.99,
    "vector_weight_rebalance": True,
    "vector_weight_unit": -1.0,
    "vector_weight_max": 4.0,
    "vector_rebalance_deadband": -1.0,
    "leaf_eval": "model",
    "leaf_rollout_r": 3,
    "leaf_vhat_tail": True,
    "gumbel_scale": 3.0,
    "max_num_considered_actions": 16,
    "deterministic": True,
    "discount": 0.997,
}
# model_size/h_cycles/l_cycles default to these and are overridable per manifest ('model').
OPERATOR_PIN_MODEL = {"model_size": "small", "h_cycles": 4, "l_cycles": 6}
# TASK_856 records these ONLY under a non-argmax decision rule — their presence means the
# battery did not run the 848 argmax read.
OPERATOR_FORBIDDEN = ("decision_rule", "decision_tau", "decision_m")
# Free-text prose eval_vector composes; recorded verbatim, never asserted.
OPERATOR_PROSE = ("selection",)
# result["panel"] fields that pin the env geometry (position sizing + encoder warmup).
PANEL_PIN = {"position_size_pct": 0.1, "warm": 60}

# Floor-artifact provenance carried into the scorecard so a cross-configuration comparison
# can never hide (runs/831_ext convention).
FLOOR_SOURCE_META_KEYS = ("tag", "db", "n_assets", "n_days", "n_decisions", "warm",
                          "position_size_pct", "discount", "value_rollout_steps",
                          "proposal_bias", "value_heuristic", "seeds")
FLOOR_OPERATOR_DESCR = ("831_ext dithered-book search grid over its own seed list "
                        "(see source_meta.seeds)")
BATTERY_OPERATOR_DESCR = "policy-argmax replay, single eval_seed"

D17_CAVEAT = ("D17: a consistency pass confers NO gate status — it completes the battery "
              "prerequisite only. Nothing here promotes anything on the 4-gate ladder.")


class ManifestError(RuntimeError):
    """Malformed manifest / unusable CLI input — aborts before anything runs."""


class WindowFailure(RuntimeError):
    """One window cannot be graded. Becomes a FAILED row (never a silent skip)."""


class OperatorPinFailure(WindowFailure):
    """The graded read was NOT the 848 operator. Poisons every window equally, so the
    battery stops instead of burning hours producing more wrong-operator rows."""


# =============================================================================
# small shared helpers
# =============================================================================

def _rel(path):
    """Stable provenance: repo-relative inside the repo, absolute outside (the
    eval_vector._display_path convention — artifacts must not embed ephemeral prefixes)."""
    p = os.path.abspath(path)
    r = os.path.relpath(p, _REPO)
    return p if r.startswith("..") else r


def _resolve(path):
    """Manifest paths are repo-relative unless absolute."""
    return path if os.path.isabs(path) else os.path.join(_REPO, path)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path, what, exc=WindowFailure):
    if not os.path.exists(path):
        raise exc(f"{what} not found: {_rel(path)}")
    try:
        with open(path) as fh:
            obj = json.load(fh)
    except (ValueError, OSError) as e:
        raise exc(f"{what} unreadable ({type(e).__name__}: {e}): {_rel(path)}")
    if not isinstance(obj, dict):
        raise exc(f"{what} is not a JSON object (got {type(obj).__name__}): {_rel(path)}")
    return obj


def _is_date(s):
    try:
        datetime.strptime(str(s), "%Y-%m-%d")
        return True
    except (ValueError, TypeError):
        return False


def _num(x):
    """JSON-safe float passthrough (None stays None).

    Rejects NaN/Infinity. `json.load` ACCEPTS the bare `NaN`/`Infinity` literals, so a floor
    artifact (or a producer result) carrying one used to flow all the way through to a
    `status: OK` row rendering "+nan" — and the emitted file was then not strict JSON, so a
    conformant consumer could not read the artifact at all. A non-finite floor is a missing
    floor: raise, and let the caller turn it into a FAILED row."""
    if x is None:
        return None
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"non-finite value {x!r} (NaN/Infinity is not a usable number here)")
    return v


def _safe_component(name):
    """A window/artifact name becomes a path component and a filename — it must not be able
    to escape the output dir ('..' matches NAME_RE but puts win_dir one level up)."""
    return bool(NAME_RE.match(name or "")) and set(name) != {"."}


def _git_describe():
    try:
        out = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=_REPO,
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None


def _as_text(x):
    if x is None:
        return ""
    return x.decode("utf-8", "replace") if isinstance(x, bytes) else str(x)


def _tail(text, n_lines=25, n_chars=4000):
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    return "\n".join(lines[-n_lines:])[-n_chars:]


def _same_path(a, b):
    return os.path.normpath(_rel(_resolve(str(a)))) == os.path.normpath(_rel(_resolve(str(b))))


def _pin_equal(got, want):
    """Type-aware equality (True == 1 must NOT pass; floats compare within JSON precision)."""
    if isinstance(want, bool) or isinstance(got, bool):
        return isinstance(got, bool) and isinstance(want, bool) and got == want
    if isinstance(want, float) or isinstance(got, float):
        try:
            return abs(float(got) - float(want)) <= 1e-12
        except (TypeError, ValueError):
            return False
    return got == want


# =============================================================================
# manifest
# =============================================================================

def load_manifest(path):
    """Read + validate the windows manifest (schema above). Raises ManifestError."""
    m = _read_json(path, "manifest", exc=ManifestError)
    if m.get("schema") != MANIFEST_SCHEMA:
        raise ManifestError(
            f"manifest {_rel(path)}: schema is {m.get('schema')!r}, expected "
            f"{MANIFEST_SCHEMA!r} — refusing to guess the layout of an unknown manifest.")
    unknown = sorted(set(m) - set(MANIFEST_KEYS))
    if unknown:
        raise ManifestError(
            f"manifest {_rel(path)}: unknown key(s) {unknown} — a mistyped key would be "
            f"silently ignored and the battery would grade under a default nobody chose. "
            f"Known keys: {list(MANIFEST_KEYS)}")
    for key in ("name", "checkpoint", "frame_provenance", "db"):
        if not isinstance(m.get(key), str) or not m[key].strip():
            raise ManifestError(f"manifest {_rel(path)}: '{key}' must be a non-empty string")
    if not _safe_component(m["name"]):
        raise ManifestError(
            f"manifest {_rel(path)}: name {m['name']!r} must match {NAME_RE.pattern} and must "
            f"not be a bare path component like '.'/'..' (it becomes the artifact filename)")
    if not isinstance(m.get("n_assets"), int) or isinstance(m.get("n_assets"), bool) \
            or m["n_assets"] < 1:
        raise ManifestError(f"manifest {_rel(path)}: 'n_assets' must be a positive int")
    m.setdefault("eval_seed", 0)
    if not isinstance(m["eval_seed"], int) or isinstance(m["eval_seed"], bool):
        raise ManifestError(f"manifest {_rel(path)}: 'eval_seed' must be an int")
    if "notes" in m and not isinstance(m["notes"], str):
        raise ManifestError(f"manifest {_rel(path)}: 'notes' must be a string when present")
    model = m.setdefault("model", {})
    if not isinstance(model, dict) or set(model) - set(OPERATOR_PIN_MODEL):
        raise ManifestError(
            f"manifest {_rel(path)}: 'model' must be an object with only "
            f"{sorted(OPERATOR_PIN_MODEL)}")
    if model.get("model_size") is not None and model["model_size"] not in MODEL_SIZES:
        raise ManifestError(
            f"manifest {_rel(path)}: model.model_size {model['model_size']!r} is not one of "
            f"{list(MODEL_SIZES)} — TrainingConfig maps an unrecognised size SILENTLY onto the "
            f"'small' architecture, so the header would advertise an architecture the numbers "
            f"were never produced under.")
    for key in ("h_cycles", "l_cycles"):
        if model.get(key) is not None and (not isinstance(model[key], int)
                                           or isinstance(model[key], bool) or model[key] < 1):
            raise ManifestError(f"manifest {_rel(path)}: model.{key} must be a positive int")
    windows = m.get("windows")
    if not isinstance(windows, list) or not windows:
        raise ManifestError(f"manifest {_rel(path)}: 'windows' must be a non-empty list")
    seen = set()
    for i, w in enumerate(windows):
        where = f"manifest {_rel(path)} windows[{i}]"
        if not isinstance(w, dict):
            raise ManifestError(f"{where}: not an object")
        unknown = sorted(set(w) - set(WINDOW_KEYS))
        if unknown:
            raise ManifestError(f"{where}: unknown key(s) {unknown} — known: {list(WINDOW_KEYS)}")
        name = w.get("name")
        if not isinstance(name, str) or not _safe_component(name):
            raise ManifestError(
                f"{where}: 'name' must match {NAME_RE.pattern} and must not be a bare path "
                f"component like '.'/'..' (it becomes an output subdirectory)")
        if name in seen:
            raise ManifestError(f"{where}: duplicate window name {name!r}")
        seen.add(name)
        if w.get("kind") not in WINDOW_KINDS:
            raise ManifestError(f"{where}: 'kind' must be one of {WINDOW_KINDS}")
        if not (_is_date(w.get("start")) and _is_date(w.get("end"))):
            raise ManifestError(f"{where}: 'start'/'end' must be YYYY-MM-DD dates")
        if str(w["start"]) >= str(w["end"]):
            raise ManifestError(f"{where}: start {w['start']} is not before end {w['end']}")
        if w["kind"] == "regime" and not w.get("floor_artifact"):
            raise ManifestError(
                f"{where}: kind 'regime' requires 'floor_artifact' (its own per-window floor, "
                f"e.g. runs/831_ext/{name}.json) — a regime window without its floor cannot be "
                f"read against anything (task scope 1/4).")
        if "floor_artifact" in w and not isinstance(w["floor_artifact"], str):
            raise ManifestError(f"{where}: 'floor_artifact' must be a string path")
        md = w.get("max_decisions")
        if md is not None and (not isinstance(md, int) or isinstance(md, bool) or md < 2):
            raise ManifestError(f"{where}: 'max_decisions' must be an int >= 2 when present")
    # Hash the manifest ONCE, here, while we are already reading it and a failure is still a
    # clean startup refusal. `build_scorecard` must never touch the filesystem: it is called on
    # the last-resort recovery path, so any I/O it does can re-raise the very exception that
    # recovery exists to survive (the manifest being moved/deleted during an hours-long battery
    # would otherwise lose the whole artifact AFTER every window had run).
    # Underscore-prefixed resolved fields. They are stripped by `manifest_roundtrip_safe()` so a
    # tool that normalizes a manifest through this loader still emits one the loader accepts
    # (the unknown-key check above would otherwise reject its own output).
    m["_sha256"] = _sha256(path)
    m["_git_describe"] = _git_describe()
    return m


def manifest_roundtrip_safe(m):
    """The manifest as authored — resolved `_`-prefixed fields removed."""
    return {k: v for k, v in m.items() if not k.startswith("_")}


# =============================================================================
# preflight — global refusal (frame / provenance / inputs)
# =============================================================================

def preflight(manifest):
    """Global input checks. Returns (frame_symbols, provenance_dict, errors[list[str]]).
    A non-empty ``errors`` REFUSES the whole run (no window is executed)."""
    errors = []
    db = _resolve(manifest["db"])
    if not os.path.exists(db):
        errors.append(f"db not found: {_rel(db)}")
    ckpt = _resolve(manifest["checkpoint"])
    if not os.path.isdir(ckpt):
        errors.append(f"checkpoint directory not found: {_rel(ckpt)}")
    frame, prov = None, {}
    prov_path = _resolve(manifest["frame_provenance"])
    try:
        prov = _read_json(prov_path, "frame provenance", exc=ManifestError)
    except ManifestError as e:
        errors.append(str(e))
        return frame, prov, errors
    frame = prov.get("frame_symbols", prov.get("symbols"))
    if not (isinstance(frame, list) and frame
            and all(isinstance(s, str) and s for s in frame)):
        errors.append(
            f"frame provenance {_rel(prov_path)}: no non-empty ordered symbol list under "
            f"'frame_symbols' (or 'symbols') — the recorded training frame is ABSENT; refusing "
            f"to guess it.")
        return None, prov, errors
    if len(set(frame)) != len(frame):
        dupes = sorted({s for s in frame if frame.count(s) > 1})
        errors.append(
            f"frame provenance {_rel(prov_path)}: DUPLICATE frame symbol(s) {dupes[:10]} — "
            f"slot k IS asset k, a repeated symbol collapses two encoder slots onto one column.")
    if len(frame) != manifest["n_assets"]:
        errors.append(
            f"frame/provenance mismatch: recorded frame has {len(frame)} symbols but the "
            f"manifest declares n_assets={manifest['n_assets']} (slot k IS asset k positionally; "
            f"a mismatched count mis-shapes the encoder kernel).")
    prov_n = prov.get("n_assets")
    if isinstance(prov_n, int) and prov_n != manifest["n_assets"]:
        errors.append(
            f"frame/provenance mismatch: provenance records n_assets={prov_n} but the manifest "
            f"declares n_assets={manifest['n_assets']}.")
    prov_ckpt = prov.get("checkpoint")
    if isinstance(prov_ckpt, str) and prov_ckpt.strip():
        if not _same_path(prov_ckpt, ckpt):
            errors.append(
                f"frame/provenance mismatch: provenance {_rel(prov_path)} was recorded for "
                f"checkpoint {prov_ckpt!r} but the manifest grades {manifest['checkpoint']!r}. "
                f"Grading a checkpoint through ANOTHER checkpoint's slot frame scrambles asset "
                f"identity — refusing.")
    return frame, prov, errors


# =============================================================================
# per-window: floor artifact + eval_vector invocation + provenance cross-checks
# =============================================================================

def load_floor_artifact(path, window):
    """A runs/831_ext-convention floor artifact: {"window":[s,e], "floor":{tier:val},
    "ceiling":{tier:val}}. Fail-loud on absence, malformation, a non-numeric/null floor
    value, or a window that disagrees with the manifest's (a floor from a DIFFERENT span is
    not this window's floor)."""
    art = _read_json(path, "floor artifact")
    for key in ("window", "floor"):
        if key not in art:
            raise WindowFailure(f"floor artifact {_rel(path)} has no '{key}' key")
    if [str(x) for x in (art["window"] or [])] != [str(window[0]), str(window[1])]:
        raise WindowFailure(
            f"floor artifact {_rel(path)} covers window {art['window']} but this manifest window "
            f"is {list(window)} — a floor from a different span is not this window's floor.")
    floor = art.get("floor") or {}
    ceiling = art.get("ceiling") or {}
    if not isinstance(floor, dict):
        raise WindowFailure(f"floor artifact {_rel(path)}: 'floor' is not an object")
    if not isinstance(ceiling, dict):
        raise WindowFailure(f"floor artifact {_rel(path)}: 'ceiling' is not an object")
    floor = {str(k): v for k, v in floor.items()}
    ceiling = {str(k): v for k, v in ceiling.items()}
    missing = [t for t in TIERS if t not in floor]
    if missing:
        raise WindowFailure(
            f"floor artifact {_rel(path)} has no floor for cost tier(s) {missing} "
            f"(need both {list(TIERS)})")
    floor_vals, ceiling_vals = {}, {}
    for t in TIERS:
        if floor[t] is None or isinstance(floor[t], bool):
            raise WindowFailure(
                f"floor artifact {_rel(path)}: tier '{t}' floor is {floor[t]!r} — a regime "
                f"window with no NUMERIC floor cannot be read against anything (a null floor "
                f"is a missing floor).")
        try:
            floor_vals[t] = float(floor[t])
        except (TypeError, ValueError):
            raise WindowFailure(
                f"floor artifact {_rel(path)}: tier '{t}' floor {floor[t]!r} is not a number.")
        cv = ceiling.get(t)
        if cv is None or isinstance(cv, bool):
            ceiling_vals[t] = None
            continue
        try:
            ceiling_vals[t] = float(cv)
        except (TypeError, ValueError):
            raise WindowFailure(
                f"floor artifact {_rel(path)}: tier '{t}' ceiling {cv!r} is not a number.")
    return {
        "path": _rel(path),
        "sha256": _sha256(path),
        "floor": floor_vals,
        "ceiling": ceiling_vals,
        # the floor was measured under its OWN universe/panel — carried IN FULL so the reader
        # can see exactly how far the comparison reaches (see config_delta below)
        "source_meta": {k: art.get(k) for k in FLOOR_SOURCE_META_KEYS},
    }


def floor_config_delta(source_meta, manifest, panel):
    """Enumerate every way the PINNED floor's measurement configuration differs from this
    battery's. The pinned 831_ext floors were measured on a different universe, position size,
    warmup, db and operator — the comparison is cross-configuration and the artifact must say
    so out loud. (Reconciling it is a planner call, not a harness call.)"""
    sm = source_meta or {}
    fields, diffs = {}, []

    def cmp(key, floor_val, battery_val, label, note=None):
        same = None
        if floor_val is not None and battery_val is not None:
            same = _pin_equal(floor_val, battery_val)
        fields[key] = {"floor": floor_val, "battery": battery_val, "same": same}
        if same is False:
            diffs.append(f"{label}: floor {floor_val} vs battery {battery_val}"
                         + (f" ({note})" if note else ""))
        elif same is None and floor_val is None:
            diffs.append(f"{label}: the floor artifact records no value — UNKNOWN, the "
                         f"comparison cannot be qualified on this axis")

    cmp("n_assets", sm.get("n_assets"), manifest.get("n_assets"), "universe breadth (n_assets)",
        "breadth is first-order for portfolio Sharpe")
    cmp("position_size_pct", _num(sm.get("position_size_pct")),
        (panel or {}).get("position_size_pct"), "position_size_pct")
    cmp("warm", sm.get("warm"), (panel or {}).get("warm"), "warmup rows (warm)")
    db_floor = sm.get("db")
    db_batt = manifest.get("db")
    same_db = None if db_floor is None else (os.path.basename(str(db_floor))
                                             == os.path.basename(str(db_batt)))
    fields["db"] = {"floor": db_floor, "battery": db_batt, "same": same_db}
    if same_db is False:
        diffs.append(f"db: floor {db_floor} vs battery {db_batt}")
    seeds = sm.get("seeds")
    n_seeds = len(seeds) if isinstance(seeds, list) else None
    floor_op = (f"{FLOOR_OPERATOR_DESCR} over {n_seeds} seed(s)" if n_seeds
                else "NOT RECORDED by the floor artifact")
    fields["operator"] = {"floor": floor_op, "battery": BATTERY_OPERATOR_DESCR,
                          "floor_n_seeds": n_seeds, "battery_n_seeds": 1, "same": False}
    diffs.append(f"operator: floor = {floor_op}; battery = {BATTERY_OPERATOR_DESCR} "
                 f"(a pinned floor is by construction not this battery's operator)")
    return {"fields": fields, "differences": diffs, "cross_config": bool(diffs)}


def child_env_block(jax_platforms):
    """Env pinned onto EVERY eval_vector subprocess (recorded verbatim in the header).

    ``JAX_PLATFORMS`` is pinned ALWAYS — to the requested value, or explicitly to "cpu" when
    none was requested. Leaving it unset let an ambient ``JAX_PLATFORMS`` select the compute
    backend while the header recorded ``jax_platforms: null``: different floats, unrecorded, and
    on the GPU it would also be a bare launch (invariant 5). The backend the child actually used
    is now always a recorded, reproducible fact rather than a property of the caller's shell."""
    env = {"XLA_PYTHON_CLIENT_MEM_FRACTION": XLA_MEM_FRACTION,
           "JAX_PLATFORMS": jax_platforms or DEFAULT_JAX_PLATFORMS}
    return env


def build_argv(manifest, window, out_dir, tag):
    argv = [sys.executable, EVAL_VECTOR,
            "--db", _resolve(manifest["db"]),
            "--start", str(window["start"]), "--end", str(window["end"]),
            "--bar-minutes", str(BAR_MINUTES),
            "--n-assets", str(manifest["n_assets"]),
            "--eval-seed", str(manifest["eval_seed"]),
            "--checkpoint", _resolve(manifest["checkpoint"]),
            "--panel-frame-from-checkpoint", _resolve(manifest["frame_provenance"]),
            *OPERATOR_ARGS,
            "--tag", tag, "--out-dir", out_dir]
    model = manifest.get("model") or {}
    for flag, key in (("--model-size", "model_size"), ("--h-cycles", "h_cycles"),
                      ("--l-cycles", "l_cycles")):
        if model.get(key) is not None:
            argv += [flag, str(model[key])]
    if window.get("max_decisions") is not None:
        argv += ["--max-decisions", str(window["max_decisions"])]
    return argv


def _expected_operator_pin(manifest):
    pin = dict(OPERATOR_PIN)
    model = manifest.get("model") or {}
    for key, default in OPERATOR_PIN_MODEL.items():
        pin[key] = model.get(key, default)
    return pin


def _check_operator(res, manifest):
    """OBSERVE the operator out of eval_vector's own result and assert the full 848 field set.
    Nothing here is an echo of our argv: every value is read back from the producer. Returns
    the list of operator fields the pin does not know about (recorded, not fatal)."""
    op = res.get("operator")
    if not isinstance(op, dict):
        raise OperatorPinFailure("result carries no 'operator' block — the read cannot be "
                                 "attributed to an operator at all")
    pin = _expected_operator_pin(manifest)
    bad = []
    for key, want in sorted(pin.items()):
        if key not in op:
            bad.append(f"operator.{key} ABSENT (pin {want!r})")
        elif not _pin_equal(op[key], want):
            bad.append(f"operator.{key} {op[key]!r} != {want!r}")
    for key in OPERATOR_FORBIDDEN:
        if key in op:
            bad.append(f"operator.{key} present ({op[key]!r}) — TASK_856 records it only under "
                       f"a NON-argmax decision rule, so this is not the 848 argmax read")
    ck = res.get("checkpoint")
    if not isinstance(ck, str) or not ck.strip():
        bad.append("result records no 'checkpoint'")
    elif not _same_path(ck, _resolve(manifest["checkpoint"])):
        bad.append(f"checkpoint {ck!r} != manifest {manifest['checkpoint']!r}")
    panel = res.get("panel") or {}
    for key, want in sorted(PANEL_PIN.items()):
        if key not in panel:
            bad.append(f"panel.{key} ABSENT (pin {want!r})")
        elif not _pin_equal(panel[key], want):
            bad.append(f"panel.{key} {panel[key]!r} != {want!r}")
    if bad:
        raise OperatorPinFailure(
            "the graded read is NOT the 848 operator of record — " + "; ".join(bad)
            + ". (The operator is observed from eval_vector's own result, not assumed from "
              "argv; a moved argparse default poisons every window, so the battery stops here.)")
    return sorted(set(op) - set(pin) - set(OPERATOR_PROSE))


def _check_result_provenance(res, manifest, window):
    """Refuse a result that does not describe the run this manifest asked for. Catches a
    stale artifact, a silently-different operator, and a frame that never took effect."""
    bad = []
    if not _same_path(res.get("db", ""), _resolve(manifest["db"])):
        bad.append(f"db {res.get('db')!r} != {manifest['db']!r}")
    if [str(x) for x in (res.get("window") or [])] != [str(window["start"]), str(window["end"])]:
        bad.append(f"window {res.get('window')} != {[window['start'], window['end']]}")
    if res.get("bar_minutes") != BAR_MINUTES:
        bad.append(f"bar_minutes {res.get('bar_minutes')} != {BAR_MINUTES}")
    if res.get("eval_seed") != manifest["eval_seed"]:
        bad.append(f"eval_seed {res.get('eval_seed')} != {manifest['eval_seed']}")
    if [int(c) for c in (res.get("cost_tiers_bp") or [])] != [int(t) for t in TIERS]:
        bad.append(f"cost_tiers_bp {res.get('cost_tiers_bp')} != {[int(t) for t in TIERS]}")
    if res.get("net_dependent") is not True:
        bad.append("net_dependent is not true (the read cannot grade a checkpoint)")
    if res.get("teacher_floor") is None:
        bad.append("teacher_floor block absent (--teacher-floor did not take effect)")
    frame_block = res.get("panel_frame")
    if not isinstance(frame_block, dict):
        bad.append("panel_frame block absent (--panel-frame-from-checkpoint did not take effect)")
    else:
        if not _same_path(frame_block.get("provenance_path", ""),
                          _resolve(manifest["frame_provenance"])):
            bad.append(f"panel_frame.provenance_path {frame_block.get('provenance_path')!r} "
                       f"!= {manifest['frame_provenance']!r}")
        if frame_block.get("n_slots") != manifest["n_assets"]:
            bad.append(f"panel_frame.n_slots {frame_block.get('n_slots')} != "
                       f"{manifest['n_assets']}")
    panel = res.get("panel") or {}
    if panel.get("n_assets") != manifest["n_assets"]:
        bad.append(f"panel.n_assets {panel.get('n_assets')} != {manifest['n_assets']}")
    if not isinstance(panel.get("n_decisions"), int):
        bad.append(f"panel.n_decisions {panel.get('n_decisions')!r} is not an int")
    for t in TIERS:
        if not isinstance((res.get("tiers") or {}).get(t), dict):
            bad.append(f"tiers['{t}'] missing")
    if bad:
        raise WindowFailure("eval_vector result provenance mismatch: " + "; ".join(bad))
    # operator is checked LAST and separately: it aborts the battery, the rest fails one window
    return _check_operator(res, manifest)


def _tier_row(tier_obj):
    tf = _num(tier_obj.get("teacher_floor_inenv"))
    sharpe = _num(tier_obj.get("sharpe_full"))
    act = tier_obj.get("activity") or {}
    bankrupt_at = tier_obj.get("bankrupt_at")
    try:
        bankrupt = bool(bankrupt_at is not None and int(bankrupt_at) >= 0)
    except (TypeError, ValueError):
        bankrupt = bool(bankrupt_at is not None)
    return {
        "sharpe": sharpe,
        "teacher_floor": tf,
        "delta_vs_teacher_floor": (None if (sharpe is None or tf is None
                                            or tier_obj.get("partial")) else sharpe - tf),
        "final_pv": _num(tier_obj.get("final_pv")),
        "bankrupt_at": bankrupt_at,
        "bankrupt": bankrupt,
        "terminated_early_at": tier_obj.get("terminated_early_at"),
        "pct_hold_agg": _num(tier_obj.get("pct_hold_agg")),
        # TASK_859 activity block, carried verbatim (keys fixed by the schema)
        "activity": {k: act.get(k) for k in
                     ("hold_pct", "trades_per_step", "n_trades_total", "n_long", "n_short",
                      "active_assets", "top5_traded")},
    }


def run_window(manifest, window, out_root, jax_platforms, verbose=True,
               timeout_s=DEFAULT_WINDOW_TIMEOUT_S):
    """Grade ONE window. Returns an OK row; raises WindowFailure with a human reason."""
    name = window["name"]
    span = (window["start"], window["end"])
    win_dir = os.path.join(out_root, "windows", name)
    os.makedirs(win_dir, exist_ok=True)
    log_path = os.path.join(win_dir, "eval.log")

    regime_floor = None
    if window.get("floor_artifact"):
        regime_floor = load_floor_artifact(_resolve(window["floor_artifact"]), span)

    tag = f"sc862_{manifest['name']}_{name}"
    # Never inherit a stale artifact from an earlier run of this window.
    for fn in sorted(os.listdir(win_dir)):
        if fn.startswith(f"results_{tag}_") and fn.endswith(".json"):
            os.remove(os.path.join(win_dir, fn))

    argv = build_argv(manifest, window, win_dir, tag)
    env = dict(os.environ)
    # Project invariant 5 — a GPU child is never launched bare. FORCED, not defaulted: an
    # ambient MEM_FRACTION must neither break the invariant nor leak into the artifact (the
    # header records exactly what the children ran under). The cgroup RAM cap belongs to the
    # safe_run wrapper the whole battery should be launched under.
    env.update(child_env_block(jax_platforms))
    if verbose:
        print(f"[862] {name}: {' '.join(argv[1:])}", flush=True)
    try:
        proc = subprocess.run(argv, cwd=_REPO, env=env, capture_output=True, text=True,
                              timeout=timeout_s)
        out, err, rc = proc.stdout, proc.stderr, proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired as e:
        out, err, rc, timed_out = _as_text(e.stdout), _as_text(e.stderr), None, True
    with open(log_path, "w") as fh:
        fh.write(out or "")
        if err:
            fh.write("\n----- stderr -----\n" + err)
        if timed_out:
            fh.write(f"\n----- TIMEOUT after {timeout_s}s -----\n")
    if timed_out:
        raise WindowFailure(
            f"eval_vector exceeded the per-window timeout of {timeout_s}s and was killed; see "
            f"{_rel(log_path)}. Raise --window-timeout if the window is genuinely this long. "
            f"Tail:\n{_tail(err or out)}")
    if rc != 0:
        raise WindowFailure(
            f"eval_vector exited {rc}; see {_rel(log_path)}. Tail:\n{_tail(err or out)}")

    found = sorted(fn for fn in os.listdir(win_dir)
                   if fn.startswith(f"results_{tag}_") and fn.endswith(".json"))
    if len(found) != 1:
        raise WindowFailure(
            f"expected exactly ONE results_{tag}_*.json in {_rel(win_dir)}, found {len(found)} "
            f"({found}) — cannot attribute the read to this window.")
    res_path = os.path.join(win_dir, found[0])
    res = _read_json(res_path, "eval_vector result")
    unpinned = _check_result_provenance(res, manifest, window)

    tiers = {t: _tier_row(res["tiers"][t]) for t in TIERS}
    net = res["tiers"][NET_TIER]
    panel = res.get("panel") or {}

    # Cross-check: when eval_vector independently loaded a 831-EXT floor for a window of this
    # name, it MUST agree with the floor artifact this manifest pinned. NOTE: in the canonical
    # configuration both sides read the SAME runs/831_ext/<name>.json, so this is tautological
    # there — it is a genuine guard only when a manifest pins a DIFFERENT floor file than the
    # one eval_vector's own 831-EXT table resolves for that window name.
    if regime_floor is not None:
        pw = (net.get("per_window") or {}).get(name)
        if isinstance(pw, dict) and pw.get("floor") is not None:
            try:
                got = float(pw["floor"])
            except (TypeError, ValueError):
                raise WindowFailure(
                    f"eval_vector's own 831-EXT floor for '{name}' is not a number: "
                    f"{pw['floor']!r}")
            want = regime_floor["floor"][NET_TIER]
            if want is None or abs(got - want) > 1e-9:
                raise WindowFailure(
                    f"floor disagreement for '{name}': the pinned artifact "
                    f"{regime_floor['path']} says net@21 floor {want}, eval_vector's own "
                    f"831-EXT load says {got}.")
        deltas = {t: (None if tiers[t]["sharpe"] is None or regime_floor["floor"][t] is None
                      or res["tiers"][t].get("partial")
                      else tiers[t]["sharpe"] - regime_floor["floor"][t]) for t in TIERS}
        regime_floor["config_delta"] = floor_config_delta(
            regime_floor.get("source_meta"), manifest, panel)
        regime_floor["delta_vs_regime_floor"] = deltas
        regime_floor["at_or_above_regime_floor_net21"] = (
            None if deltas[NET_TIER] is None else bool(deltas[NET_TIER] >= 0.0))

    frame_block = res.get("panel_frame") or {}
    return {
        "name": name,
        "kind": window["kind"],
        "status": "OK",
        "window": [span[0], span[1]],
        "n_decisions": panel.get("n_decisions"),
        "steps_completed": net.get("steps_completed"),
        "partial": bool(net.get("partial") or res["tiers"]["0"].get("partial")),
        "max_decisions": window.get("max_decisions"),
        "panel": {
            "n_rows": panel.get("n_rows"),
            "n_assets": panel.get("n_assets"),
            "n_active_slots": frame_block.get("n_active_slots"),
            "n_inactive_slots": frame_block.get("n_inactive_slots"),
            "n_partial_slots": frame_block.get("n_partial_slots"),
            "position_size_pct": _num(panel.get("position_size_pct")),
            "warm": panel.get("warm"),
        },
        # OBSERVED, verbatim — eval_vector's own operator block, not our argv echoed back
        "operator": dict(res["operator"]),
        "operator_unpinned": list(unpinned),
        "tiers": tiers,
        "regime_floor": regime_floor,
        "hold_collapse": bool((res.get("corridor") or {}).get("hold_collapse")),
        "net_dependent": bool(res.get("net_dependent")),
        "eval_artifact": _rel(res_path),
        "eval_log": _rel(log_path),
    }


# =============================================================================
# scorecard assembly + schema validation + renders
# =============================================================================

def _floor_block(row):
    """The row's regime_floor block, or {} — never raises on a malformed row."""
    rf = row.get("regime_floor") if isinstance(row, dict) else None
    return rf if isinstance(rf, dict) else {}


def _net_tier(row):
    """The row's net@21 tier object, or {} — never raises on a malformed row."""
    tiers = row.get("tiers") if isinstance(row, dict) else None
    tier = tiers.get(NET_TIER) if isinstance(tiers, dict) else None
    return tier if isinstance(tier, dict) else {}


def _cross_config(row):
    cd = _floor_block(row).get("config_delta")
    return bool(cd.get("cross_config")) if isinstance(cd, dict) else False


def _consistency(rows):
    """The DEC-2 reporting frame: >=3 of the 4 regime windows at/above THEIR window floor.
    Reported against BOTH floor bases — and they are NOT equivalent:

      * teacher floor  — on-window, same run, same universe, same operator. APPLES-TO-APPLES;
        this is the sound basis and the primary number.
      * pinned floor   — the manifest's runs/831_ext artifact, measured under a materially
        different configuration (see each row's regime_floor.config_delta). CROSS-CONFIG.

    Per D17 neither confers gate status; the form completes the battery prerequisite."""
    # Defensive throughout: a malformed row must be DEGRADED by the writer's self-check, and
    # that can only happen if assembling the header does not blow up on it first.
    regime = [r for r in rows if isinstance(r, dict) and r.get("kind") == "regime"]
    ok = [r for r in regime if r.get("status") == "OK"]
    n_art = sum(1 for r in ok if _floor_block(r).get("at_or_above_regime_floor_net21") is True)
    deltas_tf = [_net_tier(r).get("delta_vs_teacher_floor") for r in ok]
    n_tf = sum(1 for d in deltas_tf if isinstance(d, (int, float)) and d >= 0.0)
    n_reg = len(regime)
    complete = bool(regime) and len(ok) == n_reg and n_reg >= DEC2_CONSISTENCY_MIN
    dec2 = (n_reg == DEC2_N_REGIME)
    cross = any(_cross_config(r) for r in ok)
    notes = []
    if not regime:
        notes.append("no regime windows in this manifest — consistency form not applicable")
    elif len(ok) != n_reg:
        notes.append(f"{n_reg - len(ok)} of {n_reg} regime windows FAILED — the form is "
                     f"undecidable on a partial battery")
    if regime and not dec2:
        notes.append(
            f"this manifest carries {n_reg} regime window(s); the DEC-2 form is defined over "
            f"exactly {DEC2_N_REGIME} — it does NOT apply here"
            + (f", and with {n_reg} < {DEC2_CONSISTENCY_MIN} windows the "
               f"'{DEC2_CONSISTENCY_MIN} at/above' bar is unreachable by construction"
               if n_reg < DEC2_CONSISTENCY_MIN else
               f"; the counts below are '{DEC2_CONSISTENCY_MIN} of {n_reg}'"))
    if cross:
        notes.append("the pinned-floor comparison is CROSS-CONFIGURATION (see each row's "
                     "regime_floor.config_delta); meets_form_teacher_floor is the "
                     "apples-to-apples number")
    notes.append(D17_CAVEAT)
    return {
        "form": (f">= {DEC2_CONSISTENCY_MIN} of {DEC2_N_REGIME} regime windows at/above their "
                 f"window floor (net@21) — the DEC-2 form" if dec2 else
                 f">= {DEC2_CONSISTENCY_MIN} at/above their window floor (net@21) out of the "
                 f"{n_reg} regime window(s) in this manifest — NOT the DEC-2 form "
                 f"({DEC2_CONSISTENCY_MIN} of {DEC2_N_REGIME})"),
        "dec2_applicable": dec2,
        "floor_bases": {
            "teacher": ("on-window teacher floor computed by the SAME run — same universe, "
                        "panel, operator and costs (apples-to-apples)"),
            "pinned": ("per-window floor artifact pinned by the manifest (row "
                       "regime_floor.path, canonically runs/831_ext/<name>.json) — measured "
                       "under a DIFFERENT configuration; see regime_floor.config_delta"),
        },
        "n_regime_windows": n_reg,
        "n_regime_ok": len(ok),
        "n_at_or_above_regime_floor": n_art,
        "n_at_or_above_teacher_floor": n_tf,
        "required": DEC2_CONSISTENCY_MIN,
        "primary_basis": "meets_form_teacher_floor",
        "meets_form_teacher_floor": (bool(n_tf >= DEC2_CONSISTENCY_MIN) if complete else None),
        "meets_form_pinned_floor_cross_config": (bool(n_art >= DEC2_CONSISTENCY_MIN)
                                                 if complete else None),
        "d17": D17_CAVEAT,
        "note": "; ".join(notes),
    }


def build_scorecard(manifest, manifest_path, rows, prov, global_errors, jax_platforms,
                    child_env=None, timeout_s=DEFAULT_WINDOW_TIMEOUT_S, self_check=None):
    n_failed = sum(1 for r in rows if not isinstance(r, dict) or r.get("status") != "OK")
    checked = self_check or {"ok": True, "errors": [], "degraded_rows": []}
    cross = any(_cross_config(r) for r in rows
                if isinstance(r, dict) and r.get("status") == "OK")
    return {
        "schema": SCORECARD_SCHEMA,
        "generated_by": "experiments/835_stage4_eval/scorecard.py (TASK_862)",
        "manifest": {
            "path": _rel(manifest_path),
            "name": manifest["name"],
            "schema": MANIFEST_SCHEMA,
            # Pre-computed in load_manifest — NOT hashed here. build_scorecard runs on the
            # last-resort recovery path and must be filesystem-free (see load_manifest).
            "sha256": manifest.get("_sha256"),
        },
        "checkpoint": _rel(_resolve(manifest["checkpoint"])),
        "frame_provenance": {
            "path": _rel(_resolve(manifest["frame_provenance"])),
            "n_frame_symbols": len(prov.get("frame_symbols", prov.get("symbols")) or []),
            "meta": {k: v for k, v in sorted(prov.items())
                     if k not in ("frame_symbols", "symbols")},
        },
        "db": _rel(_resolve(manifest["db"])),
        "n_assets": manifest["n_assets"],
        "bar_minutes": BAR_MINUTES,
        "eval_seed": manifest["eval_seed"],
        "model": dict(manifest.get("model") or {}),
        "jax_platforms": jax_platforms,
        "child_env": dict(child_env or {}),
        "window_timeout_s": int(timeout_s),
        "capped_windows": [w["name"] for w in manifest["windows"]
                           if w.get("max_decisions") is not None],
        "notes": manifest.get("notes"),
        "operator": {
            "mode": OPERATOR_PIN["mode"],
            "teacher_floor": True,
            "vector_weight_rebalance": True,
            "panel_frame_from_checkpoint": True,
            "decision_rule": "argmax",
            "cost_tiers_bp": [int(t) for t in TIERS],
            "argv_template": " ".join(["eval_vector.py", "--db <db>", "--start <s>", "--end <e>",
                                       f"--bar-minutes {BAR_MINUTES}", "--n-assets <n>",
                                       "--eval-seed <seed>", "--checkpoint <ckpt>",
                                       "--panel-frame-from-checkpoint <frame>", *OPERATOR_ARGS]),
            "source": OPERATOR_SOURCE,
            "pin": _expected_operator_pin(manifest),
            "panel_pin": dict(PANEL_PIN),
            "verification": (
                "each window's result['operator'] is OBSERVED out of eval_vector's own JSON, "
                "copied verbatim into the row, and asserted field-by-field against 'pin' "
                "(plus panel position_size_pct/warm against 'panel_pin' and the result's own "
                "checkpoint against the manifest). Fields present in the result but absent "
                "from 'pin' are listed per row under operator_unpinned. A disagreement FAILS "
                "the window and ABORTS the battery."),
        },
        # Resolved ONCE in load_manifest, not here: build_scorecard runs on the recovery path and
        # must not fork. (_self_check re-builds once per row, so this also stops a degraded
        # 7-window battery shelling out to git 9 times.)
        "git_describe": manifest.get("_git_describe"),
        "status": ("FAILED" if (n_failed or global_errors or not checked["ok"]) else "OK"),
        "n_windows": len(rows),
        "n_ok": len(rows) - n_failed,
        "n_failed": n_failed,
        "global_errors": list(global_errors),
        "self_check": checked,
        "cross_config_floor": bool(cross),
        "consistency": _consistency(rows),
        "windows": rows,
    }


def validate_scorecard(sc):
    """Schema checker for ``imprint_scorecard/v1`` (raises ValueError). The writer self-checks
    with it; the fixture smoke asserts on it; a gate-ladder submission can cite it."""
    def need(obj, key, types, where):
        if key not in obj:
            raise ValueError(f"{where}: missing key '{key}'")
        if not isinstance(obj[key], types):
            raise ValueError(f"{where}: '{key}' has type {type(obj[key]).__name__}, "
                             f"expected {types}")

    if not isinstance(sc, dict):
        raise ValueError("scorecard is not an object")
    if sc.get("schema") != SCORECARD_SCHEMA:
        raise ValueError(f"scorecard schema {sc.get('schema')!r} != {SCORECARD_SCHEMA!r}")
    for key, types in (("generated_by", str), ("manifest", dict), ("checkpoint", str),
                       ("frame_provenance", dict), ("db", str), ("n_assets", int),
                       ("bar_minutes", int), ("eval_seed", int), ("model", dict),
                       ("child_env", dict), ("window_timeout_s", int),
                       ("capped_windows", list), ("operator", dict), ("status", str),
                       ("n_windows", int), ("n_ok", int), ("n_failed", int),
                       ("global_errors", list), ("self_check", dict),
                       ("cross_config_floor", bool), ("consistency", dict), ("windows", list)):
        need(sc, key, types, "scorecard")
    for key in ("path", "name", "schema", "sha256"):
        need(sc["manifest"], key, str, "scorecard.manifest")
    for key in ("pin", "panel_pin"):
        need(sc["operator"], key, dict, "scorecard.operator")
    for key in ("verification", "source", "argv_template"):
        need(sc["operator"], key, str, "scorecard.operator")
    need(sc["self_check"], "ok", bool, "scorecard.self_check")
    need(sc["self_check"], "errors", list, "scorecard.self_check")
    need(sc["self_check"], "degraded_rows", list, "scorecard.self_check")
    for key in ("form", "note", "d17", "primary_basis"):
        need(sc["consistency"], key, str, "scorecard.consistency")
    need(sc["consistency"], "floor_bases", dict, "scorecard.consistency")
    need(sc["consistency"], "dec2_applicable", bool, "scorecard.consistency")
    for key in ("n_regime_windows", "n_regime_ok", "n_at_or_above_regime_floor",
                "n_at_or_above_teacher_floor", "required"):
        need(sc["consistency"], key, int, "scorecard.consistency")
    for key in ("meets_form_teacher_floor", "meets_form_pinned_floor_cross_config"):
        if key not in sc["consistency"]:
            raise ValueError(f"scorecard.consistency: missing key '{key}'")
        if not isinstance(sc["consistency"][key], (bool, type(None))):
            raise ValueError(f"scorecard.consistency: '{key}' must be a bool or null")
    if sc["status"] not in ("OK", "FAILED"):
        raise ValueError(f"scorecard.status {sc['status']!r} not in ('OK','FAILED')")
    if sc["n_windows"] != len(sc["windows"]):
        raise ValueError("scorecard.n_windows disagrees with len(windows)")
    n_failed = sum(1 for r in sc["windows"] if r.get("status") != "OK")
    if sc["n_failed"] != n_failed or sc["n_ok"] != len(sc["windows"]) - n_failed:
        raise ValueError("scorecard.n_ok/n_failed disagree with the window rows")
    clean = (n_failed == 0 and not sc["global_errors"] and sc["self_check"]["ok"])
    if (sc["status"] == "OK") != clean:
        raise ValueError("scorecard.status disagrees with the failed-row / global-error / "
                         "self-check state")
    names = set()
    for i, r in enumerate(sc["windows"]):
        where = f"scorecard.windows[{i}]"
        for key, types in (("name", str), ("kind", str), ("status", str), ("window", list)):
            need(r, key, types, where)
        if r["name"] in names:
            raise ValueError(f"{where}: duplicate window name {r['name']!r}")
        names.add(r["name"])
        if r["kind"] not in WINDOW_KINDS:
            raise ValueError(f"{where}: kind {r['kind']!r} not in {WINDOW_KINDS}")
        if len(r["window"]) != 2:
            raise ValueError(f"{where}: 'window' must be [start, end]")
        if r["status"] == "FAILED":
            need(r, "reason", str, where)
            continue
        if r["status"] != "OK":
            raise ValueError(f"{where}: status {r['status']!r} not in ('OK','FAILED')")
        for key, types in (("n_decisions", int), ("partial", bool), ("panel", dict),
                           ("operator", dict), ("operator_unpinned", list),
                           ("tiers", dict), ("hold_collapse", bool), ("net_dependent", bool),
                           ("eval_artifact", str), ("eval_log", str)):
            need(r, key, types, where)
        if "max_decisions" not in r:
            raise ValueError(f"{where}: missing key 'max_decisions' (may be null)")
        if not isinstance(r["max_decisions"], (int, type(None))):
            raise ValueError(f"{where}: 'max_decisions' must be an int or null")
        if set(r["tiers"]) != set(TIERS):
            raise ValueError(f"{where}: tiers keys {sorted(r['tiers'])} != {sorted(TIERS)}")
        for t in TIERS:
            tw = f"{where}.tiers['{t}']"
            for key in ("sharpe", "teacher_floor", "delta_vs_teacher_floor", "final_pv"):
                if key not in r["tiers"][t]:
                    raise ValueError(f"{tw}: missing key '{key}'")
                if not isinstance(r["tiers"][t][key], (float, int, type(None))):
                    raise ValueError(f"{tw}: '{key}' must be a number or null")
            need(r["tiers"][t], "bankrupt", bool, tw)
            need(r["tiers"][t], "activity", dict, tw)
            for key in ("hold_pct", "trades_per_step", "n_trades_total", "n_long", "n_short",
                        "active_assets", "top5_traded"):
                if key not in r["tiers"][t]["activity"]:
                    raise ValueError(f"{tw}.activity: missing key '{key}'")
        if r["kind"] == "regime":
            if not isinstance(r.get("regime_floor"), dict):
                raise ValueError(f"{where}: kind 'regime' must carry a regime_floor block")
            for key in ("path", "sha256", "floor", "ceiling", "source_meta", "config_delta",
                        "delta_vs_regime_floor"):
                if key not in r["regime_floor"]:
                    raise ValueError(f"{where}.regime_floor: missing key '{key}'")
            for key in ("differences", "fields", "cross_config"):
                if key not in r["regime_floor"]["config_delta"]:
                    raise ValueError(f"{where}.regime_floor.config_delta: missing key '{key}'")
        elif "regime_floor" not in r:
            raise ValueError(f"{where}: missing key 'regime_floor' (may be null)")
    return True


def _fmt(x, nd=3):
    return "n/a" if x is None else f"{float(x):+.{nd}f}"


def _yn(v):
    return "n/a" if v is None else ("yes" if v else "no")


def _row_flags(r):
    flags = []
    if r.get("partial"):
        flags.append("PARTIAL")
    if r.get("max_decisions") is not None:
        flags.append(f"CAPPED@{r['max_decisions']}")
    if r.get("hold_collapse"):
        flags.append("HOLD-COLLAPSE")
    if _cross_config(r):
        flags.append("CROSS-CFG")
    if r.get("operator_unpinned"):
        flags.append("UNPINNED-OP")
    tiers = r.get("tiers") if isinstance(r.get("tiers"), dict) else {}
    for t in TIERS:
        tier = tiers.get(t)
        if isinstance(tier, dict) and tier.get("bankrupt"):
            flags.append(f"BANKRUPT@{t}")
    return ",".join(flags) if flags else "-"


def render_md(sc):
    L = []
    L.append(f"# Imprint benchmark scorecard — {sc['manifest']['name']}")
    L.append("")
    L.append(f"- **status**: {sc['status']}  ({sc['n_ok']} OK / {sc['n_failed']} FAILED "
             f"of {sc['n_windows']} windows)")
    L.append(f"- **checkpoint**: `{sc['checkpoint']}`")
    L.append(f"- **frame provenance**: `{sc['frame_provenance']['path']}` "
             f"({sc['frame_provenance']['n_frame_symbols']} slots)")
    L.append(f"- **db**: `{sc['db']}` · n_assets {sc['n_assets']} · bar_minutes "
             f"{sc['bar_minutes']} · eval_seed {sc['eval_seed']}")
    L.append(f"- **operator**: policy-argmax replay (decision-rule argmax) + on-window teacher "
             f"floor + weight-rebalance + fixed-frame panel; tiers "
             f"{sc['operator']['cost_tiers_bp']} bp ({sc['operator']['source']}). "
             f"OBSERVED per window from eval_vector's own result and asserted against the pin.")
    model = sc["model"] or {}
    L.append(f"- **model**: {model if model else 'eval_vector defaults (small / H4 / L6)'} "
             f"· window timeout {sc['window_timeout_s']}s · child env {sc['child_env']}")
    L.append(f"- **git**: `{sc['git_describe']}` · manifest sha256 "
             f"`{sc['manifest']['sha256'][:12]}`")
    L.append(f"- **schema**: `{sc['schema']}`")
    if sc["capped_windows"]:
        L.append(f"- **CAPPED**: {sc['capped_windows']} ran under a `max_decisions` cap — "
                 f"NOT full windows, not citable as a read of record")
    if sc["cross_config_floor"]:
        L.append("- **CROSS-CONFIG FLOORS**: at least one pinned floor was measured under a "
                 "different configuration than this battery — see *Configuration delta* below. "
                 "The apples-to-apples number is the on-window **teacher** floor.")
    if not sc["self_check"]["ok"]:
        L.append(f"- **SELF-CHECK FAILED**: {sc['self_check']['errors']} "
                 f"(degraded rows: {sc['self_check']['degraded_rows']})")
    if sc["notes"]:
        L.append(f"- **notes**: {sc['notes']}")
    if sc["global_errors"]:
        L.append("")
        L.append("## REFUSED — global preflight failed")
        L.append("")
        for e in sc["global_errors"]:
            L.append(f"- {e}")
    L.append("")
    L.append("## Windows")
    L.append("")
    L.append("| window | kind | span | dec | flags | net@21 | floor(on-win) | Δ | gross | "
             "floor(gross) | Δ(gross) | regime floor (cross-cfg) | Δ vs regime | trades/step | "
             "bankrupt | status |")
    L.append("|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|:-:|")
    for r in sc["windows"]:
        if r["status"] != "OK":
            L.append(f"| {r['name']} | {r['kind']} | {r['window'][0]}..{r['window'][1]} | "
                     f"| | | | | | | | | | | | **FAILED** |")
            continue
        net, gross = r["tiers"][NET_TIER], r["tiers"]["0"]
        rf = r.get("regime_floor") or {}
        rf_v = (rf.get("floor") or {}).get(NET_TIER)
        rf_d = (rf.get("delta_vs_regime_floor") or {}).get(NET_TIER)
        L.append(
            f"| {r['name']} | {r['kind']} | {r['window'][0]}..{r['window'][1]} | "
            f"{r['n_decisions']} | {_row_flags(r)} | {_fmt(net['sharpe'])} | "
            f"{_fmt(net['teacher_floor'])} | {_fmt(net['delta_vs_teacher_floor'])} | "
            f"{_fmt(gross['sharpe'])} | {_fmt(gross['teacher_floor'])} | "
            f"{_fmt(gross['delta_vs_teacher_floor'])} | {_fmt(rf_v)} | {_fmt(rf_d)} | "
            f"{_fmt(net['activity'].get('trades_per_step'), 2)} | "
            f"{'YES' if net['bankrupt'] else 'no'} | OK |")
    L.append("")
    L.append("Flags: PARTIAL = the rollout did not complete (deltas suppressed) · CAPPED@n = "
             "ran under a `max_decisions` cap · HOLD-COLLAPSE = the TASK_821 collapse signature "
             "· CROSS-CFG = the pinned floor is cross-configuration · UNPINNED-OP = the result "
             "carries operator fields the 848 pin does not know about.")

    deltas = [(r["name"], r["regime_floor"]["config_delta"], r["regime_floor"].get("path"),
               r["regime_floor"].get("source_meta"))
              for r in sc["windows"]
              if r.get("status") == "OK" and isinstance(r.get("regime_floor"), dict)
              and r["regime_floor"].get("config_delta", {}).get("cross_config")]
    if deltas:
        L.append("")
        L.append("## Configuration delta — the pinned floors are NOT same-configuration")
        L.append("")
        L.append("`Δ vs regime` above compares this battery against floors measured under the "
                 "configuration below. Read it as a cross-configuration reference point, not as "
                 "a like-for-like margin; the like-for-like number is `Δ` (on-window teacher "
                 "floor, same run, same universe).")
        for name, cd, path, sm in deltas:
            L.append("")
            L.append(f"### {name} — floor `{path}`")
            L.append("")
            for d in cd["differences"]:
                L.append(f"- {d}")
            L.append(f"- floor source_meta: `{json.dumps(sm, sort_keys=True)}`")

    failed = [r for r in sc["windows"] if r["status"] != "OK"]
    if failed:
        L.append("")
        L.append("## FAILED windows")
        for r in failed:
            L.append("")
            L.append(f"### {r['name']} ({r['kind']}, {r['window'][0]}..{r['window'][1]})")
            L.append("")
            L.append("```")
            L.append(r.get("reason", "(no reason recorded)"))
            L.append("```")
    c = sc["consistency"]
    L.append("")
    L.append("## Consistency (reporting frame — NO gate status, D17)")
    L.append("")
    L.append(f"- form: {c['form']}")
    L.append(f"- floor basis — teacher (sound): {c['floor_bases']['teacher']}")
    L.append(f"- floor basis — pinned (cross-config): {c['floor_bases']['pinned']}")
    L.append(f"- regime windows: {c['n_regime_ok']}/{c['n_regime_windows']} graded; "
             f"{c['n_at_or_above_teacher_floor']} at/above their on-window teacher floor, "
             f"{c['n_at_or_above_regime_floor']} at/above their pinned (cross-config) floor "
             f"(required {c['required']})")
    L.append(f"- **meets_form_teacher_floor (PRIMARY, apples-to-apples)**: "
             f"{_yn(c['meets_form_teacher_floor'])}")
    L.append(f"- meets_form_pinned_floor_cross_config (secondary, cross-configuration): "
             f"{_yn(c['meets_form_pinned_floor_cross_config'])}")
    L.append(f"- {c['note']}")
    L.append("")
    return "\n".join(L)


def _render_md_safe(sc):
    try:
        return render_md(sc)
    except Exception as e:                                      # noqa: BLE001
        return ("# Imprint benchmark scorecard — RENDER FAILED\n\n"
                f"The JSON artifact was written; the markdown render raised "
                f"`{type(e).__name__}: {e}`.\n\n```\n{traceback.format_exc()[-2000:]}\n```\n")


def print_summary(sc):
    print("=" * 84)
    print(f"IMPRINT SCORECARD ({sc['manifest']['name']}) — {sc['status']} "
          f"[{sc['n_ok']} OK / {sc['n_failed']} FAILED]")
    print(f"checkpoint {sc['checkpoint']} · frame {sc['frame_provenance']['path']} "
          f"· db {sc['db']}")
    for e in sc["global_errors"]:
        print(f"  REFUSED: {e}")
    if not sc["self_check"]["ok"]:
        print(f"  SELF-CHECK FAILED: {sc['self_check']['errors']}")
    for r in sc["windows"]:
        if r["status"] != "OK":
            first = (r.get("reason") or "").splitlines()
            print(f"  {r['name']:<12} FAILED — {first[0] if first else ''}")
            continue
        net = r["tiers"][NET_TIER]
        rf = (r.get("regime_floor") or {}).get("delta_vs_regime_floor") or {}
        print(f"  {r['name']:<12} {r['kind']:<11} n={r['n_decisions']:<5} "
              f"net@21 {_fmt(net['sharpe'])} floor {_fmt(net['teacher_floor'])} "
              f"Δ {_fmt(net['delta_vs_teacher_floor'])} "
              f"Δregime {_fmt(rf.get(NET_TIER))} [{_row_flags(r)}]")
    c = sc["consistency"]
    print(f"Consistency — {c['form']}:")
    print(f"  teacher floor (PRIMARY): {c['n_at_or_above_teacher_floor']}/"
          f"{c['n_regime_windows']} at/above (required {c['required']}) → "
          f"meets_form_teacher_floor={_yn(c['meets_form_teacher_floor'])}")
    print(f"  pinned floor (cross-config): {c['n_at_or_above_regime_floor']}/"
          f"{c['n_regime_windows']} at/above → "
          f"meets_form_pinned_floor_cross_config="
          f"{_yn(c['meets_form_pinned_floor_cross_config'])}")
    print(f"  {c['note']}")
    print("=" * 84)


# =============================================================================
# entry point
# =============================================================================

def _failed_row(w, out_dir, reason):
    row = {"name": w["name"], "kind": w["kind"], "status": "FAILED",
           "window": [w["start"], w["end"]], "reason": reason}
    log_path = os.path.join(out_dir, "windows", w["name"], "eval.log")
    if os.path.exists(log_path):
        row["eval_log"] = _rel(log_path)
    return row


def _self_check(sc, manifest, manifest_path, out_dir, prov, global_errors, jax_platforms,
                child_env, timeout_s):
    """Validate the assembled scorecard. A schema violation must never destroy the artifact:
    a header violation is recorded and the file is written anyway; a ROW violation degrades
    THAT row to FAILED (its numbers remain on disk in the row's eval artifact) and the
    scorecard is reassembled. Exit code stays non-zero either way."""
    try:
        validate_scorecard(sc)
        return sc
    except ValueError as first:
        errors = [str(first)]

    wmap = {w["name"]: w for w in manifest["windows"]}

    def _rebuild(rows, checked):
        return build_scorecard(manifest, manifest_path, rows, prov, global_errors,
                               jax_platforms, child_env=child_env, timeout_s=timeout_s,
                               self_check=checked)

    # Is the HEADER itself schema-valid? (probe with no rows)
    probe = _rebuild([], {"ok": True, "errors": [], "degraded_rows": []})
    try:
        validate_scorecard(probe)
    except ValueError as e:
        errors.append(f"header: {e}")
        sc["self_check"] = {"ok": False, "errors": errors, "degraded_rows": []}
        sc["status"] = "FAILED"
        return sc

    rows, degraded = [], []
    for r in sc["windows"]:
        one = _rebuild([r], {"ok": True, "errors": [], "degraded_rows": []})
        try:
            validate_scorecard(one)
            rows.append(r)
        except ValueError as e:
            name = r.get("name")
            degraded.append(name if isinstance(name, str) else str(name))
            errors.append(f"windows[{name!r}]: {e}")
            w = wmap.get(name)
            reason = (f"row degraded by the writer's schema self-check ({e}). The window's own "
                      f"numbers are on disk at "
                      f"{r.get('eval_artifact', '(no eval artifact recorded)')}.")
            rows.append(_failed_row(w, out_dir, reason) if w else
                        {"name": str(name), "kind": "regime", "status": "FAILED",
                         "window": ["?", "?"], "reason": reason})
    out = _rebuild(rows, {"ok": False, "errors": errors, "degraded_rows": degraded})
    try:
        validate_scorecard(out)
    except ValueError as e:
        out["self_check"]["errors"].append(f"post-degrade: {e}")
        out["status"] = "FAILED"
    return out


def run_battery(manifest_path, out_dir, jax_platforms=None, verbose=True,
                timeout_s=DEFAULT_WINDOW_TIMEOUT_S):
    """Run the whole battery. Returns (scorecard dict, json_path, md_path). Exit code is the
    caller's: non-zero iff scorecard['status'] != 'OK'.

    NOTHING a window can throw is allowed to destroy the artifact: every per-window exception
    becomes a FAILED row, and the writer always emits both files."""
    manifest = load_manifest(manifest_path)
    os.makedirs(out_dir, exist_ok=True)
    frame, prov, global_errors = preflight(manifest)
    child_env = child_env_block(jax_platforms)

    rows = []
    if global_errors:
        # REFUSE: nothing runs, but every window still gets an explicit FAILED row so the
        # artifact can never read as a partial success.
        reason = "REFUSED (global preflight): " + "; ".join(global_errors)
        for w in manifest["windows"]:
            rows.append({"name": w["name"], "kind": w["kind"], "status": "FAILED",
                         "window": [w["start"], w["end"]], "reason": reason})
    else:
        abort = None
        for w in manifest["windows"]:
            if abort is not None:
                rows.append(_failed_row(w, out_dir, abort))
                if verbose:
                    print(f"[862] {w['name']}: FAILED — not run ({abort.splitlines()[0]})")
                continue
            try:
                rows.append(run_window(manifest, w, out_dir, jax_platforms, verbose=verbose,
                                       timeout_s=timeout_s))
                continue
            except OperatorPinFailure as e:
                reason = str(e)
                abort = (f"battery ABORTED at window '{w['name']}': the operator of record was "
                         f"not honoured, so every remaining window would be graded under the "
                         f"same wrong operator. {reason}")
            except WindowFailure as e:
                reason = str(e)
            except Exception as e:                              # noqa: BLE001
                # ANY other failure (OSError on the out-dir, a schema drift in eval_vector's
                # own JSON, a bug here) is a FAILED ROW — never a lost battery.
                reason = (f"unexpected {type(e).__name__} while grading this window: {e}\n"
                          f"{_tail(traceback.format_exc(), n_lines=12)}")
            rows.append(_failed_row(w, out_dir, reason))
            if verbose:
                print(f"[862] {w['name']}: FAILED — {reason.splitlines()[0]}")

    try:
        sc = build_scorecard(manifest, manifest_path, rows, prov, global_errors, jax_platforms,
                             child_env=child_env, timeout_s=timeout_s)
        sc = _self_check(sc, manifest, manifest_path, out_dir, prov, global_errors,
                         jax_platforms, child_env, timeout_s)
    except Exception as e:                                      # noqa: BLE001
        # LAST RESORT: even the assembler blowing up must not swallow the battery. Rebuild
        # from the (already validated) manifest with every row degraded to FAILED — the
        # per-window numbers stay on disk under <out>/windows/<name>/.
        reason = (f"the scorecard writer raised {type(e).__name__}: {e} — this row was "
                  f"degraded to FAILED; its numbers, if the window ran, remain on disk under "
                  f"{_rel(out_dir)}/windows/{{name}}/")
        try:
            sc = build_scorecard(
                manifest, manifest_path,
                [_failed_row(w, out_dir, reason.replace("{name}", w["name"]))
                 for w in manifest["windows"]],
                prov, global_errors, jax_platforms, child_env=child_env, timeout_s=timeout_s,
                self_check={"ok": False,
                            "errors": [f"{type(e).__name__}: {e}",
                                       _tail(traceback.format_exc(), n_lines=12)],
                            "degraded_rows": [w["name"] for w in manifest["windows"]]})
        except Exception as e2:                                 # noqa: BLE001
            # FLOOR OF THE CONTRACT: even the recovery assembler failing must leave an artifact
            # on disk that says so. Hand-built from primitives only — no helper that could raise.
            sc = {
                "schema": SCORECARD_SCHEMA,
                "generated_by": "experiments/835_stage4_eval/scorecard.py (TASK_862)",
                "status": "FAILED",
                "n_windows": len(manifest.get("windows") or []),
                "n_ok": 0,
                "windows": [{"name": str((w or {}).get("name", "?")), "status": "FAILED",
                             "reason": "scorecard assembly AND its recovery path both raised; "
                                       "per-window eval artifacts, if any, remain on disk under "
                                       "windows/<name>/"}
                            for w in (manifest.get("windows") or [])],
                "consistency": {"d17": D17_CAVEAT},
                # Provenance MUST survive into the tombstone: without it you cannot tell from the
                # file WHICH battery died. All of these are already-validated primitives.
                "manifest": {"path": _rel(manifest_path), "name": manifest.get("name"),
                             "schema": MANIFEST_SCHEMA, "sha256": manifest.get("_sha256")},
                "checkpoint": str(manifest.get("checkpoint")),
                "db": str(manifest.get("db")),
                "frame_provenance": {"path": str(manifest.get("frame_provenance"))},
                "n_failed": len(manifest.get("windows") or []),
                "self_check": {"ok": False, "errors": [
                    f"primary: {type(e).__name__}: {e}",
                    f"recovery: {type(e2).__name__}: {e2}",
                    _tail(traceback.format_exc(), n_lines=12)]},
            }
    json_path = os.path.join(out_dir, f"scorecard_{manifest['name']}.json")
    md_path = os.path.join(out_dir, f"scorecard_{manifest['name']}.md")
    # SERIALIZE BEFORE OPENING ANYTHING, and write via tmp+os.replace. Writing straight into the
    # target left a truncated, unparseable JSON and NO md when serialization failed part-way
    # (e.g. a future key holding a set/datetime/ndarray) — strictly worse than no artifact,
    # because a consumer can half-read it. If the card cannot be serialized we fall back to the
    # tombstone rather than emitting a partial file.
    try:
        json_text = json.dumps(sc, indent=2, default=float, allow_nan=False) + "\n"
        md_text = _render_md_safe(sc)
    except Exception as e3:                                     # noqa: BLE001
        sc = {
            "schema": SCORECARD_SCHEMA, "status": "FAILED",
            "generated_by": "experiments/835_stage4_eval/scorecard.py (TASK_862)",
            "manifest": {"path": _rel(manifest_path), "name": manifest.get("name"),
                         "schema": MANIFEST_SCHEMA, "sha256": manifest.get("_sha256")},
            "checkpoint": str(manifest.get("checkpoint")), "db": str(manifest.get("db")),
            "n_windows": len(manifest.get("windows") or []), "n_ok": 0,
            "n_failed": len(manifest.get("windows") or []),
            "windows": [{"name": str((w or {}).get("name", "?")), "status": "FAILED",
                         "reason": "the assembled scorecard was not JSON-serializable; per-window "
                                   "eval artifacts, if any, remain under windows/<name>/"}
                        for w in (manifest.get("windows") or [])],
            "consistency": {"d17": D17_CAVEAT},
            "self_check": {"ok": False,
                           "errors": [f"serialization: {type(e3).__name__}: {e3}"]},
        }
        json_text = json.dumps(sc, indent=2, default=str, allow_nan=False) + "\n"
        md_text = _render_md_safe(sc)
    for path, text in ((json_path, json_text), (md_path, md_text)):
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            fh.write(text)
        os.replace(tmp, path)
    return sc, json_path, md_path


def main():
    ap = argparse.ArgumentParser(
        description="TASK_862 — imprint benchmark scorecard (battery → one artifact)")
    ap.add_argument("--manifest", required=True,
                    help="windows manifest JSON (schema imprint_scorecard_manifest/v1; the "
                         "canonical battery: experiments/835_stage4_eval/"
                         "scorecard_manifest_imprint.json)")
    ap.add_argument("--out-dir", default=None,
                    help="output dir (default: runs/862_scorecard)")
    ap.add_argument("--jax-platforms", default=None, dest="jax_platforms",
                    help="JAX_PLATFORMS for the per-window eval_vector subprocesses. It is "
                         "always FORCED into the child env (default 'cpu' when this flag is "
                         "absent) so an ambient value cannot silently pick the backend; the "
                         "effective value is recorded in the artifact's child_env. Children are "
                         "also forced to XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 — which is NOT a cap "
                         "(eval_vector sets PREALLOCATE=false), so launch the battery under "
                         "safe_run for the cgroup RAM cap (invariant 5).")
    ap.add_argument("--window-timeout", type=int, default=DEFAULT_WINDOW_TIMEOUT_S,
                    dest="window_timeout",
                    help=f"per-window eval_vector timeout in seconds (default "
                         f"{DEFAULT_WINDOW_TIMEOUT_S}); on expiry the child is killed and the "
                         f"window becomes a FAILED row instead of hanging the battery")
    ap.add_argument("--quiet", action="store_true", help="suppress per-window progress lines")
    args = ap.parse_args()
    if args.window_timeout < 1:
        print("[862] --window-timeout must be >= 1 second", file=sys.stderr)
        return 2

    out_dir = args.out_dir or os.path.join(_REPO, "runs", "862_scorecard")
    try:
        sc, json_path, md_path = run_battery(
            os.path.abspath(args.manifest), os.path.abspath(out_dir),
            jax_platforms=args.jax_platforms, verbose=not args.quiet,
            timeout_s=args.window_timeout)
    except ManifestError as e:
        print(f"[862] REFUSED — bad manifest: {e}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"[862] REFUSED — cannot use the output dir / inputs: "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        return 2
    print_summary(sc)
    print(f"[862] wrote {_rel(json_path)}")
    print(f"[862] wrote {_rel(md_path)}")
    return 0 if sc["status"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
