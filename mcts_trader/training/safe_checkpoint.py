"""
Safe Checkpoint Manager for Curriculum Training.

Provides intelligent checkpoint management that:
- Only saves checkpoints when performance improves
- Detects performance degradation and triggers rollback
- Maintains N best checkpoints for recovery
- Tracks rolling performance metrics

Usage:
    manager = SafeCheckpointManager(
        checkpoint_dir='checkpoints/level_5',
        metric='avg_reward',
        patience=50,
        rollback_threshold=0.5
    )

    for epoch in training:
        metrics = train_epoch()

        if manager.should_save(metrics['avg_reward']):
            manager.save_checkpoint(params, opt_state, epoch, metrics['avg_reward'])

        if manager.should_rollback(metrics['avg_reward']):
            params, opt_state = manager.load_best_checkpoint()

D64 SCOPE — THE `--apply` FREEZE DOES NOT COVER THIS MODULE (TASK_881D resume, 2026-08-26)
------------------------------------------------------------------------------------------
D64 froze ``tools/ckpt_gc.py``'s ``--apply``: the offline CLI now refuses to delete anything
unless an explicitly-named override is spelled in full with an exact literal value, because the
manifest COVERAGE predicate has no AUTHORITY BINDING yet (a well-formed ``covers:``/``protect:``
line in ANY ``experiments/<id>/PROTECTED.txt`` grants coverage over ANY run dir it names, with no
check that ``<id>`` has any relationship to that run dir; TASK_881E is the binding that closes it).

**That gate lives in ``ckpt_gc.py``'s ``main()``/argparse path and NOWHERE ELSE, so NONE of it
applies here.** ``_gc_module()`` below loads ``ckpt_gc.py`` by path and this module calls
``load_manifests`` / ``scan_referenced_paths`` / ``plan`` / ``collect`` DIRECTLY; ``train_jax_fast.py``
then calls ``prune_run_dir_by_score(..., apply=True)`` unconditionally on every score-retention
save. So the IN-RUN deleter:

* is exposed to the identical unbound manifest-coverage-authority hole D64 documents;
* runs UNATTENDED inside every live training process with ``ckpt_retention == 'score'``;
* has NO override, NO acknowledgement and no way for an operator to be prompted about the risk.

This is DISCLOSED, not fixed, and the asymmetry is deliberate: D64's ruling shipped TASK_881D
rather than blocking training, and a freeze that made every live training run start refusing to
prune would be a worse outcome than the disclosed gap. Do not read ``_D64_APPLY_FREEZE_REFUSAL``,
or D64's own undifferentiated phrase "this row is what first makes it gate an irreversible
deleter", as covering deletion generally — it covers the offline CLI only. Gating this path is a
recommendation to the planner (TASK_881E), not something this row did.

WHAT *IS* SHARED, so the disclosure is not read as "the in-run pruner has no protection at all":
the no-coverage refusal itself (``resolve_coverage`` in ``prune_run_dir_by_score``), the
``protect:``/``covers:`` manifest semantics, ``plan()``'s retention arithmetic, the stale-scan
refusal, and the force-rescan-before-delete rule. What is NOT shared is the D64 freeze.
"""

import json
import logging
import math
import os
import shutil
import time
import uuid
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Sequence
from dataclasses import dataclass, field

# One id per training PROCESS. Written on every score row so a resumed run (which restarts
# `epoch` at 0 while writing into the same scores.jsonl) stays distinguishable in the audit
# trail from the run it resumed — see TASK_876 M8.
_RUN_SESSION_ID = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"


# ============================================================================================
# TASK_881 — in-run score-based retention. The offline `tools/ckpt_gc.py` is the single source
# of the retention/protection policy (manifest resolver, `_is_protected`, `plan()`); this module
# loads it BY PATH (not a package import — the tool is deliberately standalone/stdlib-only so it
# can be copied to and run on `gpu-server` in isolation) and re-uses its functions directly.
#
# WHAT THAT DOES AND DOES NOT GUARANTEE (corrected, TASK_881D cycle 2 — the earlier wording
# here claimed "the in-run pruner and the offline tool can never disagree about the same tree",
# which was FALSIFIED BY MEASUREMENT and is not restated). Sharing `plan()`/`_is_protected`
# makes the two deleters apply the same POLICY to the same INPUTS. It says nothing about the
# inputs being the same: this module TTL-caches the protection scan and the offline tool
# re-reads from disk on every invocation, so a cached scan up to `ttl_seconds` old could —
# and did — make the in-run pruner DELETE a checkpoint the offline tool protected in the same
# second (repro: manifest dropped mid-run, cache warm, `apply=True`, `stale=False`). The fix is
# in `get_protection`/`prune_run_dir_by_score`: A CALL THAT MAY DELETE NEVER READS THE CACHE.
# So the accurate claim is narrower and is the one this module now enforces:
#   * a DELETING call (`apply=True`) re-scans the tree unconditionally first, and therefore
#     cannot disagree with the offline tool about the same tree at the same instant;
#   * a NON-deleting call (`apply=False`, the dry-run/reporting path) may still serve a
#     protection set up to `ttl_seconds` old and may therefore print a "would delete" line the
#     offline tool would not — a reporting difference only, on a path that removes nothing.
# ============================================================================================

_GC_MODULE = None


def _gc_module():
    """Load ``tools/ckpt_gc.py`` as the single source of the retention/protection policy.

    Returns ``None`` if it cannot be loaded — score-based retention CANNOT WORK without it, and
    the caller must not silently continue with an unbounded/unprotected save; see
    ``retention_policy_available()``, which the CLI checks at startup.
    """
    global _GC_MODULE
    if _GC_MODULE is None:
        import importlib.util
        p = Path(__file__).resolve().parents[2] / "tools" / "ckpt_gc.py"
        try:
            spec = importlib.util.spec_from_file_location("_task881_ckpt_gc", p)
            if spec is None or spec.loader is None:
                raise ImportError(f"no loader for {p}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _GC_MODULE = mod
        except Exception as e:
            logging.error(f"[ckpt-retention] cannot load the retention policy from {p}: {e} "
                          f"— score-based retention is UNAVAILABLE")
            return None
    return _GC_MODULE


def retention_policy_available() -> Tuple[bool, str]:
    """Can score-based retention actually run in this deployment? Called at CLI startup — see
    ``mcts_trader/training/cli.py``. If the policy module is missing (a partial deploy, a
    copied-out trainer), the caller must fall back to ``recency`` rather than discover the gap
    only when the disk is full."""
    p = Path(__file__).resolve().parents[2] / "tools" / "ckpt_gc.py"
    if _gc_module() is None:
        return False, f"retention policy module not loadable: {p}"
    return True, str(p)


# TASK_881 mechanical fix (D32/D33, model item 5): the protect set used to be computed ONCE at
# launch. `SERVER.md` documents mid-run `experiments/` SPEC/manifest drops as NORMAL workflow,
# so a launch-time snapshot silently disagrees with reality the moment a manifest changes — a
# checkpoint newly protected by a manifest written after the process started got zero
# protection until the next restart. A TTL cache re-scans lazily instead, bounding staleness to
# `ttl_seconds` rather than "until the process exits". Module-level and NOT thread-safe by
# design (this package has one in-run pruner call site per process).
_PROTECTION_CACHE: Dict[str, Any] = {"named": set(), "protect": set(), "missing": set(),
                                     "ts": 0.0, "attempted": False, "stale": False}
_DEFAULT_PROTECTION_TTL_SECONDS = 300.0


def collect_protection(roots: Sequence[str] = ("jax_ckpt_fast", "runs"),
                       scan_dirs: Sequence[str] = ("experiments",)
                       ) -> Optional[Tuple[set, set, set]]:
    """One manifest + advisory scan, for IN-RUN use. -> ``(named, protect, missing)`` or
    ``None`` on failure.

    ``named`` is the gate-(i) coverage set (manifest ``protect:``/``covers:`` paths only —
    never the advisory scan, exactly like ``tools/ckpt_gc.py``'s own ``resolve_coverage``).
    ``protect`` is the full deletion-protect set (manifest ``protect:`` paths, plus the
    advisory-scan rescues — never subtracting). ``missing`` is the set of named paths that do
    not exist (TASK_881 MAJOR E); it is carried through so the in-run pruner withdraws coverage
    for exactly the run dirs the offline tool does — a dangling manifest entry must not mean
    one deleter refuses while the other prunes. (TASK_881D: sourced from ``manifests.dangling``
    — ``tools/ckpt_gc.py``'s 881I-shipped ``ManifestSet`` names this field ``dangling``, not
    ``missing``; this function's own return-tuple name is unchanged, only its source field.)

    Returns ``None`` on ANY failure (module unloadable, incomplete manifest scan, incomplete
    advisory scan) rather than an empty result — the distinction is load-bearing: an empty
    result and a FAILED scan must never be confused, or a transient failure reads as "nothing
    to protect" (see ``get_protection``, which is the only thing that should call this)."""
    gc = _gc_module()
    if gc is None:
        logging.error("[ckpt-retention] protection model UNAVAILABLE: the retention policy "
                      "module could not be loaded — reporting SCAN FAILED, not an empty set")
        return None
    scan_dirs = list(scan_dirs)
    try:
        manifests = gc.load_manifests(scan_dirs)
    except gc.ManifestScanIncomplete as e:
        logging.error(f"[ckpt-retention] manifest scan INCOMPLETE ({e}) — reporting failure so "
                      f"the caller keeps its last-known-good protection set rather than "
                      f"pruning against a set that may be missing real protection")
        return None
    try:
        scanned = gc.scan_referenced_paths(scan_dirs, [str(r) for r in roots])
    except gc.ProtectScanIncomplete as e:
        logging.error(f"[ckpt-retention] advisory scan INCOMPLETE ({e}) — reporting failure")
        return None
    return (set(manifests.protect) | set(manifests.covers), set(manifests.protect) | scanned,
            set(manifests.dangling))


def get_protection(roots: Sequence[str] = ("jax_ckpt_fast", "runs"),
                   scan_dirs: Sequence[str] = ("experiments",),
                   ttl_seconds: float = _DEFAULT_PROTECTION_TTL_SECONDS,
                   force_rescan: bool = False
                   ) -> Tuple[set, set, set]:
    """TTL-cached wrapper over ``collect_protection`` — call this at every save site under
    ``--ckpt-retention score`` instead of snapshotting the result once. Re-scans when
    ``force_rescan`` is set, when nothing has been attempted yet, or when the cache is older
    than ``ttl_seconds``; otherwise returns the cached ``(named, protect, missing)`` triple.

    ``force_rescan=True`` IS THE ONLY THING A DELETER MAY USE, and it is not a tuning knob
    (TASK_881D cycle-2 BLOCKER). A TTL bounds how STALE the set may be; it does not make a
    stale set safe, and ``protection_cache_is_stale()`` does not help here because it only
    fires when a scan FAILED — a scan that SUCCEEDED 299 s ago against a different tree reports
    ``stale=False`` and reads as authoritative. Measured end-to-end: cache warmed at the first
    save of a run; an operator then drops ``experiments/e1/PROTECTED.txt`` with
    ``protect: runs/r1/snapshot_ep10`` (the mid-run manifest drop ``SERVER.md`` documents as
    NORMAL workflow, and the exact case this TTL was introduced for); in the same second the
    offline ``tools/ckpt_gc.py`` printed ``keeping 4`` with ep10 protected while
    ``prune_run_dir_by_score(..., apply=True)`` printed ``DELETE runs/r1/snapshot_ep10`` and
    deleted it, ``stale=False``, ``had_failure=False``, no warning. The protect set is the ONE
    set that may never be served from cache to an irreversible, unsupervised, per-save deleter,
    so ``prune_run_dir_by_score`` passes ``force_rescan=apply`` and the cache survives only for
    the read-only (dry-run/reporting) path. Cost of the unconditional re-scan, measured on the
    real repo: 1.694 s total (``load_manifests`` 0.038 s over 448 dirs / 60 manifests +
    ``scan_referenced_paths`` 1.656 s), against saves that are minutes apart.

    Only a scan that succeeds COMPLETELY may replace the cache. On failure the last-known-good
    sets are KEPT (never silently emptied) and ``protection_cache_is_stale()`` becomes True,
    which ``prune_run_dir_by_score`` treats as load-bearing — it REFUSES to delete anything
    while the cache is stale, rather than delete against a set that might be missing real
    protection.

    THE BLAST RADIUS IS THE WHOLE PROCESS, NOT THE ONE RUN DIR BEING SAVED (TASK_881D, D45
    planner note 4 — documented here because the effect was previously undocumented, not new).
    ``collect_protection`` scans ALL of ``scan_dirs`` (default: every ``experiments/<id>/``) in
    ONE pass and this cache is a single module-level tuple, not one per run dir. So a SINGLE
    unreadable manifest or advisory doc — one dangling symlink, one malformed ``PROTECTED.txt``
    line, ANYWHERE under ``scan_dirs``, in an experiment that has nothing to do with the run
    currently being saved — makes this call return the STALE last-known-good set, which makes
    ``prune_run_dir_by_score`` refuse to prune EVERY run dir (for the rest of ``ttl_seconds`` on
    the dry-run path, and on EVERY deleting call for as long as the bad document is there, since
    a deleting call re-scans unconditionally and so re-hits the same failure)
    (repro: a healthy, fully-covered ``runs/healthy_run`` with its own clean ``PROTECTED.txt``
    was refused solely because an unrelated ``experiments/exp_broken/`` carried one dangling
    symlink). Independently judged CORRECT (fail-safe is the right DIRECTION — refusing to
    prune beats pruning against a possibly-incomplete protection set) but the blast radius is
    global by ASSERTION, not necessity: neither ``load_manifests`` nor ``scan_referenced_paths``
    attributes a failure to the specific run dir(s) downstream of it, so there is no cheaper
    unit than "everything" to invalidate. Scoping this to just the affected experiment would
    need both offline-tool scans to report PER-DIRECTORY partial success instead of one
    tree-wide incomplete/complete verdict — a structural change to `tools/ckpt_gc.py`'s
    ingestion contract, not undertaken by this row (TASK_881D scope item 5: document, don't
    reopen 881I's settled ingestion to chase it).
    """
    now = time.time()
    if (force_rescan or (not _PROTECTION_CACHE["attempted"])
            or (now - _PROTECTION_CACHE["ts"] >= ttl_seconds)):
        got = collect_protection(roots, scan_dirs)
        _PROTECTION_CACHE["ts"] = now
        _PROTECTION_CACHE["attempted"] = True
        if got is None:
            _PROTECTION_CACHE["stale"] = True
            logging.error(
                f"[ckpt-retention] protection scan FAILED — KEEPING the last-known-good "
                f"protection set ({len(_PROTECTION_CACHE['named'])} covered path(s), "
                f"{len(_PROTECTION_CACHE['protect'])} protected path(s)) rather than treating "
                f"the failure as 'nothing to protect'. It may be STALE; the next call after "
                f"{ttl_seconds:.0f}s retries.")
        else:
            (_PROTECTION_CACHE["named"], _PROTECTION_CACHE["protect"],
             _PROTECTION_CACHE["missing"]) = got
            _PROTECTION_CACHE["stale"] = False
    return (_PROTECTION_CACHE["named"], _PROTECTION_CACHE["protect"],
            _PROTECTION_CACHE["missing"])


def protection_cache_is_stale() -> bool:
    """True when the most recent protection-scan attempt FAILED (or was INCOMPLETE) and the
    sets ``get_protection`` returns are therefore a retained last-known-good pair.

    LOAD-BEARING, not advisory: ``prune_run_dir_by_score`` REFUSES to delete anything while
    this is True. A stale set means a manifest added since the last successful scan may be
    unprotected, and the only safe move for a deleter is not to delete. PROCESS-WIDE, not
    per-run-dir: see ``get_protection``'s docstring — one bad manifest/advisory doc anywhere
    under ``scan_dirs`` disables pruning for every run dir, not just the one it names."""
    return bool(_PROTECTION_CACHE["stale"])


def _reset_protection_cache_for_test() -> None:
    """Test-only: force the next ``get_protection()`` call to re-scan."""
    _PROTECTION_CACHE["named"] = set()
    _PROTECTION_CACHE["protect"] = set()
    _PROTECTION_CACHE["missing"] = set()
    _PROTECTION_CACHE["attempted"] = False
    _PROTECTION_CACHE["stale"] = False
    _PROTECTION_CACHE["ts"] = 0.0


def prune_run_dir_by_score(run_dir, roots: Sequence[str] = ("jax_ckpt_fast", "runs"),
                           scan_dirs: Sequence[str] = ("experiments",),
                           keep_best: int = 3, keep_latest: int = 1, keep_unscored: int = 2,
                           under_floor: bool = False,
                           ttl_seconds: float = _DEFAULT_PROTECTION_TTL_SECONDS,
                           apply: bool = True) -> Tuple[List[str], bool]:
    """Prune ONE run dir using the EXACT SAME policy ``tools/ckpt_gc.py``'s offline ``plan()``
    uses — same function, same manifest/advisory sources.

    WHERE THE INPUTS COME FROM, which is where the two deleters could differ (TASK_881D cycle 2;
    the previous sentence here claimed they "can never disagree about the same tree" and that
    claim was falsified by measurement — see ``get_protection``):

    * ``apply=True`` (this call MAY DELETE) — the protection scan is re-run UNCONDITIONALLY
      first (``force_rescan=True``), exactly as the offline tool re-reads from disk on every
      invocation. On the same tree at the same instant the two therefore see the same protect
      set and reach the same verdict. If that fresh scan FAILS, ``protection_cache_is_stale()``
      is True and this function REFUSES to delete anything rather than fall back to the
      last-known-good set.
    * ``apply=False`` (dry run / reporting, deletes nothing) — the TTL cache is used, because
      this path removes no bytes and re-scanning it on every save of a multi-day process buys
      nothing. Its "would delete" list may therefore be up to ``ttl_seconds`` behind the tree.

    (TASK_881 gate iii is about the two deleters sharing ONE ``plan()``; that is still true and
    is not what moved. What moved is the input freshness on the deleting path.)

    D64: THIS DELETER IS NOT BEHIND ``ckpt_gc.py``'s ``--apply`` FREEZE (see this module's own
    docstring for the full statement). The freeze is implemented in ``ckpt_gc.py``'s
    ``main()``/argparse path — ``_d64_apply_refusal_reason`` — and this function never goes through
    argparse: it calls ``gc.load_manifests`` / ``gc.resolve_coverage`` / ``gc.plan`` /
    ``gc.collect`` directly, and ``train_jax_fast.py`` calls it with a literal ``apply=True`` on
    every score-retention save. So an operator who has read D64 and believes checkpoint deletion is
    frozen is WRONG about this call. It is left ungated on purpose (freezing it would make live
    training stop pruning, which D64's ruling explicitly preferred not to do by shipping TASK_881D
    rather than blocking training); closing it properly is TASK_881E's authority binding.

    Returns ``(messages, had_failure)``. Never raises for a DESIGNED refusal (unresolved run
    dir, unavailable policy module, a stale protection cache) or for a failed delete — those are
    reported in ``messages``, and ``had_failure`` tells the caller whether a delete FAILED
    (rename ok, ``rmtree`` failed — orphaned bytes, recorded as a FAILED prune, never a silent
    success — TASK_881 scope item 4 / gate v) so it can decide whether that should make the
    training process exit non-zero.
    """
    msgs: List[str] = []
    gc = _gc_module()
    if gc is None:
        msgs.append(f"  [CKPT] score-based retention UNAVAILABLE for {run_dir}: the retention "
                    f"policy module could not be loaded — nothing pruned this call.")
        return msgs, False
    # `force_rescan=apply` — a call that may DELETE never reads the TTL cache (cycle-2 BLOCKER,
    # see `get_protection`). Deliberately keyed off `apply` and not off a separate flag: there
    # must be no way to reach the deleting branch below with a cached protect set.
    named, protect, missing = get_protection(roots, scan_dirs, ttl_seconds, force_rescan=apply)
    if protection_cache_is_stale():
        msgs.append(f"  [CKPT] protection scan is STALE (the last scan attempt failed or none "
                    f"has succeeded yet) — REFUSING to prune {run_dir} this call rather than "
                    f"delete against a possibly-incomplete protection set.")
        return msgs, False
    coverage = gc.resolve_coverage([str(run_dir)], named, missing)
    if not coverage.get(str(run_dir)):
        msgs.append(f"  [CKPT] REFUSE {run_dir}: no manifest declares coverage for it, or a "
                    f"manifest entry pointing into it names a path that does not exist — add "
                    f"or fix a `protect:`/`covers:` entry in an experiments/<id>/PROTECTED.txt "
                    f"or SPEC.md (see tools/ckpt_gc.py's module docstring). Nothing pruned.")
        return msgs, False
    # TASK_881 MAJOR D — resolve IDENTICALLY to the offline tool. `gc.collect` realpaths an
    # explicitly-passed root, so an honest spelling (`jax_ckpt_fast/run_a`, where the root is a
    # symlink to /mnt/scratch) resolves the same way in BOTH deleters instead of the
    # offline one refusing while this one — the unsupervised, per-save one — deletes. The
    # CONFIGURED roots are handed to `collect` as `allowed_roots` so a run dir that resolves
    # OUTSIDE them (a run moved off-disk and symlinked back into `runs/`) is still an ESCAPE and
    # still refuses: the offline tool cannot see such a directory at all, and this pruner must
    # not delete what the supervised tool cannot even list.
    walk_root, was_resolved = gc.root_resolution(str(run_dir))
    if was_resolved:
        msgs.append(f"  [CKPT] run dir {run_dir} -> {walk_root} (resolved)")
    try:
        found = gc.collect([str(run_dir)], allowed_roots=[str(r) for r in roots])
    except gc.DiscoveryScanIncomplete as e:
        msgs.append(f"  [CKPT] REFUSE {run_dir}: checkpoint discovery REFUSED ({e}) — nothing "
                    f"is pruned. Point the run at a real directory inside a configured root.")
        return msgs, False
    ckpts = found.get(walk_root, found.get(str(run_dir), []))
    if not ckpts:
        return msgs, False
    deletions, summary, index_rows, score_refused = gc.plan(
        {walk_root: ckpts}, lambda rd: gc._load_scores(rd, None, "*.log"),
        keep_best, keep_latest, under_floor, keep_unscored, protect)
    for s in summary:
        msgs.append(f"  [CKPT] {s}")
    if walk_root in score_refused:
        # A refusal is SIDE-EFFECT-FREE (TASK_881 BLOCKER A): no index write, no prune history.
        # `checkpoints_index.json` is a SCORE SOURCE, and rewriting it from a refusal is what
        # destroyed the peak checkpoint's recorded score in the offline tool's repro.
        return msgs, False
    if not apply:
        for p, size, reason in deletions:
            msgs.append(f"  [CKPT] would delete {p} ({size/1e6:.0f} MB) — {reason}")
        return msgs, False

    had_failure = False
    deleted: set = set()
    rmtree_failed: set = set()
    events: List[Dict[str, Any]] = []
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for path, size, reason in deletions:
        ok, why = gc._safe_target(path, [walk_root] + [str(r) for r in roots])
        if not ok:
            msgs.append(f"  [CKPT] REFUSE {path}: {why}")
            continue
        if gc._is_protected(path, protect):
            msgs.append(f"  [CKPT] PROTECTED {path}: manifest / advisory rescue")
            continue
        msgs.append(f"  [CKPT] DELETE {path} ({size/1e6:.0f} MB) — {reason}")
        trash = f"{path}.gc-trash-{os.getpid()}"
        try:
            os.rename(path, trash)      # atomic: a half-deleted ckpt can never look loadable
        except OSError as e:
            msgs.append(f"  [CKPT] ERROR deleting {path}: rename failed: {e}")
            had_failure = True
            continue
        try:
            shutil.rmtree(trash)
        except OSError as e:
            msgs.append(f"  [CKPT] ERROR deleting {path}: rmtree of {trash} failed: {e} — "
                        f"orphaned, bytes NOT reclaimed (recorded as a FAILED prune)")
            rmtree_failed.add(os.path.realpath(path))
            had_failure = True
            continue
        deleted.add(os.path.realpath(path))
        events.append({"ts": stamp, "path": os.path.realpath(path), "bytes": size,
                       "reason": reason, "tool": "in-run pruner"})
    gc.append_prune_history(walk_root, events)
    planned_rows = index_rows.get(walk_root)
    if planned_rows is None:
        # Nothing was planned for this run dir (it was refused upstream) — write NOTHING. An
        # empty row list here used to be written straight over `checkpoints_index.json`, which
        # is a SCORE SOURCE: the refusal would erase the scores it exists to defend.
        msgs.append(f"  [CKPT] no plan for {walk_root} — index left untouched")
    else:
        rows = gc.finalize_index_rows(planned_rows, deleted, rmtree_failed)
        gc.write_run_index(walk_root, rows, {"generated": stamp, "keep_best": keep_best,
                                             "keep_latest": keep_latest, "applied": True,
                                             "source": "in-run pruner"})
    if had_failure:
        msgs.append(f"  [CKPT] FAILED: {len(rmtree_failed)} delete(s) left orphaned bytes in "
                    f"{run_dir} — see ERROR lines above")
    return msgs, had_failure


def append_checkpoint_score(run_dir, epoch: int, score: float, metric: str,
                            path=None, step: Optional[int] = None,
                            note: Optional[str] = None) -> None:
    """Append one score row for a just-saved checkpoint (TASK_876).

    Writes one JSON object per line to ``<run_dir>/scores.jsonl``. ``run_dir`` is the
    checkpoint's PARENT directory as ``tools/ckpt_gc.py``'s directory walk sees it (the
    directory that holds the ``snapshot_ep<N>`` / ``checkpoint_<step>`` siblings), not the
    checkpoint itself.

    **What the CURRENTLY SHIPPED reader can actually do with these rows — measured, not
    assumed.** ``tools/ckpt_gc.py`` is UNCHANGED by TASK_876 and is EPOCH-KEYED: its
    ``_load_scores`` builds ``{int(row["epoch"]): float(row["score"])}`` and ignores ``path``
    and ``step`` entirely, while ``plan()`` looks a checkpoint up by ``_epoch_of(dir)``, which
    parses the digits out of the DIRECTORY BASENAME. So of the three names this codebase
    writes:

    * ``snapshot_ep<N>``      — round-trips. ``N`` really is the epoch, so the row resolves.
    * ``checkpoint_<step>``   — does NOT resolve. flax names the dir by
      ``state.step_count`` (= iterations_per_epoch x epoch), so ``_epoch_of`` yields the STEP
      and the epoch-keyed lookup misses. Reads as UNSCORED.
    * ``checkpoint_epoch_<N>`` (``SafeCheckpointManager``, incl. ``checkpoint_golden``) — does
      NOT resolve either: ``_epoch_of``'s regex needs digits immediately after
      ``checkpoint_`` and returns ``None``, and ``_is_ckpt_dir`` does not match the name.
      Reads as UNSCORED.

    UNSCORED is the FAIL-SAFE direction: ``plan()`` never score-prunes an unscored checkpoint,
    so an unresolved row cannot itself cause a deletion. Precisely (do not round this up to
    "unscored is immortal"): unscored dirs stay recency-prunable beyond ``--keep-unscored``
    when the disk is under ``--min-free-gb``, exactly as they were before TASK_876. Making the
    reader resolve by ``path``/``step`` (a path-first reader) is **TASK_881**, the checkpoint
    protection model. This row deliberately ships ZERO deleter surface.

    ``path`` and ``step`` are therefore recorded for that FUTURE path-first reader, and for
    the audit trail today — the row format is forward-compatible, not yet fully consumed.
    Always pass ``path`` when the on-disk name is known; ``epoch`` is the key the shipped
    reader uses, and is meaningful for ``snapshot_ep<N>`` only.

    A non-finite score is NOT written as a score (TASK_876 M6): NaN/Inf make a ranking sort
    non-transitive and input-order dependent, and a NaN can take a kept slot from a
    legitimately-scored checkpoint. The row is still written for the audit trail, with
    ``score: null``; ``_load_scores`` drops such a row (``float(None)`` raises and the line is
    skipped), so it reads as UNSCORED — see the fail-safe note above.

    This is pure logging: append-only, best-effort, and MUST NEVER interrupt a training run.
    A failure here (missing dir, disk full, bad permissions) is swallowed and warned — the
    checkpoint save it documents has already happened, and a missing score row only means the
    GC treats that checkpoint as unscored, which is its documented fail-safe.
    """
    try:
        d = Path(run_dir)
        d.mkdir(parents=True, exist_ok=True)
        row: Dict[str, Any] = {
            "epoch": int(epoch),
            "score": None,
            "metric": str(metric),
            "path": (os.path.abspath(str(path)) if path is not None else None),
            "step": (int(step) if step is not None else None),
            "ts": time.time(),
            "session": _RUN_SESSION_ID,
        }
        if note:
            row["note"] = str(note)
        try:
            s = float(score)
        except (TypeError, ValueError):
            s = float("nan")
        if math.isfinite(s):
            row["score"] = s
        else:
            row["score_invalid"] = repr(score)
            logging.warning(
                f"append_checkpoint_score: non-finite score {score!r} for epoch {epoch} in "
                f"{run_dir} — recording as UNSCORED (the checkpoint is never score-pruned "
                f"and never 'best'; it stays recency-prunable under ckpt_gc's disk floor)")
        with open(d / "scores.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
    except Exception as e:
        logging.warning(f"append_checkpoint_score: failed to record score for {run_dir}: {e}")


@dataclass
class CheckpointRecord:
    """Record of a saved checkpoint."""
    epoch: int
    metric: float
    path: Path
    timestamp: float = 0.0


class SafeCheckpointManager:
    """
    Intelligent checkpoint management for curriculum training.

    Features:
    - Only saves on performance improvement (rolling average)
    - Auto-rollback on significant degradation
    - Maintains best-N checkpoints
    - Tracks rolling performance metrics
    - Optional golden checkpoint preservation
    """

    def __init__(
        self,
        checkpoint_dir: str,
        metric_name: str = 'avg_reward',
        patience: int = 50,
        rollback_threshold: float = 0.5,
        keep_best_n: int = 3,
        rolling_window: int = 20,
        min_improvement: float = 0.01,
        save_golden: bool = True,
    ):
        """
        Args:
            checkpoint_dir: Directory to store checkpoints
            metric_name: Name of metric to optimize (for logging)
            patience: Epochs to wait before considering rollback
            rollback_threshold: Performance drop to trigger rollback
            keep_best_n: Number of best checkpoints to keep
            rolling_window: Window size for moving average
            min_improvement: Minimum improvement to save checkpoint
            save_golden: Save initial "golden" checkpoint as fallback
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.metric_name = metric_name
        self.patience = patience
        self.rollback_threshold = rollback_threshold
        self.keep_best_n = keep_best_n
        self.rolling_window = rolling_window
        self.min_improvement = min_improvement
        self.save_golden = save_golden

        # State tracking
        self.best_metric: float = float('-inf')
        self.best_checkpoints: List[CheckpointRecord] = []
        self.metric_history: List[float] = []
        self.epochs_since_improvement: int = 0
        self.rollback_count: int = 0
        self.golden_checkpoint: Optional[Path] = None
        self.level_start_metric: Optional[float] = None

        # Load existing state if available
        self._load_state()

        logging.info(f"SafeCheckpointManager initialized: {checkpoint_dir}")
        logging.info(f"  patience={patience}, threshold={rollback_threshold}, keep_best_n={keep_best_n}")

    def _load_state(self):
        """Load manager state from checkpoint directory."""
        state_file = self.checkpoint_dir / 'manager_state.json'
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                self.best_metric = state.get('best_metric', float('-inf'))
                self.metric_history = state.get('metric_history', [])
                self.epochs_since_improvement = state.get('epochs_since_improvement', 0)
                self.rollback_count = state.get('rollback_count', 0)
                self.level_start_metric = state.get('level_start_metric')

                # Rebuild checkpoint records
                for ckpt in state.get('best_checkpoints', []):
                    path = Path(ckpt['path'])
                    if path.exists():
                        self.best_checkpoints.append(CheckpointRecord(
                            epoch=ckpt['epoch'],
                            metric=ckpt['metric'],
                            path=path,
                            timestamp=ckpt.get('timestamp', 0)
                        ))

                if state.get('golden_checkpoint'):
                    golden_path = Path(state['golden_checkpoint'])
                    if golden_path.exists():
                        self.golden_checkpoint = golden_path

                logging.info(f"Loaded manager state: best_metric={self.best_metric:.3f}, rollbacks={self.rollback_count}")
            except Exception as e:
                logging.warning(f"Failed to load manager state: {e}")

    def _save_state(self):
        """Save manager state to checkpoint directory."""
        state_file = self.checkpoint_dir / 'manager_state.json'
        state = {
            'best_metric': self.best_metric,
            'metric_history': self.metric_history[-200:],  # Keep last 200
            'epochs_since_improvement': self.epochs_since_improvement,
            'rollback_count': self.rollback_count,
            'level_start_metric': self.level_start_metric,
            'best_checkpoints': [
                {
                    'epoch': ckpt.epoch,
                    'metric': ckpt.metric,
                    'path': str(ckpt.path),
                    'timestamp': ckpt.timestamp
                }
                for ckpt in self.best_checkpoints
            ],
            'golden_checkpoint': str(self.golden_checkpoint) if self.golden_checkpoint else None
        }

        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def get_rolling_average(self) -> float:
        """Calculate rolling average of recent metrics."""
        if not self.metric_history:
            return float('-inf')

        window = min(len(self.metric_history), self.rolling_window)
        return float(np.mean(self.metric_history[-window:]))

    def record_metric(self, metric: float):
        """Record a metric value (call every epoch)."""
        self.metric_history.append(metric)

        # Track level start metric for absolute degradation detection
        if self.level_start_metric is None and len(self.metric_history) >= self.rolling_window:
            self.level_start_metric = self.get_rolling_average()

    def should_save(self, current_metric: float) -> bool:
        """
        Determine if we should save a checkpoint.

        Saves when rolling average improves beyond min_improvement.
        """
        self.record_metric(current_metric)

        if len(self.metric_history) < self.rolling_window:
            # Not enough data - save first checkpoint as golden
            if self.save_golden and self.golden_checkpoint is None and len(self.metric_history) == 1:
                return True
            return False

        rolling_avg = self.get_rolling_average()
        improvement = rolling_avg - self.best_metric

        if improvement > self.min_improvement:
            self.best_metric = rolling_avg
            self.epochs_since_improvement = 0
            logging.info(f"Performance improved: {rolling_avg:.3f} (improvement: +{improvement:.3f})")
            return True
        else:
            self.epochs_since_improvement += 1
            return False

    def should_rollback(self, current_metric: float) -> bool:
        """
        Determine if we should rollback to last good checkpoint.

        Triggers when:
        1. Rolling average dropped significantly from best
        2. We've waited at least `patience` epochs
        """
        if len(self.metric_history) < self.rolling_window:
            return False

        if not self.best_checkpoints and not self.golden_checkpoint:
            return False  # Nothing to rollback to

        rolling_avg = self.get_rolling_average()
        degradation = self.best_metric - rolling_avg

        # Also check absolute degradation from level start
        absolute_degradation = 0.0
        if self.level_start_metric is not None:
            absolute_degradation = self.level_start_metric - rolling_avg

        # Rollback conditions:
        should_rollback = (
            degradation > self.rollback_threshold and
            self.epochs_since_improvement >= self.patience
        ) or (
            # Absolute degradation: model is worse than when level started
            absolute_degradation > self.rollback_threshold * 2
        )

        if should_rollback:
            logging.warning("=" * 60)
            logging.warning("ROLLBACK TRIGGERED")
            logging.warning("=" * 60)
            logging.warning(f"  Best metric:     {self.best_metric:.3f}")
            logging.warning(f"  Current rolling: {rolling_avg:.3f}")
            logging.warning(f"  Degradation:     {degradation:.3f} (threshold: {self.rollback_threshold})")
            logging.warning(f"  Epochs waiting:  {self.epochs_since_improvement} (patience: {self.patience})")
            if self.level_start_metric is not None:
                logging.warning(f"  Level start:     {self.level_start_metric:.3f}")
                logging.warning(f"  Absolute drop:   {absolute_degradation:.3f}")

        return should_rollback

    def save_checkpoint(
        self,
        checkpoint_data: Dict[str, Any],
        epoch: int,
        metric: float
    ) -> Path:
        """
        Save a checkpoint.

        Args:
            checkpoint_data: Dictionary of data to save (params, optimizer, etc.)
            epoch: Current epoch number
            metric: Current metric value

        Returns:
            Path to saved checkpoint
        """
        import time

        # Handle golden checkpoint (first save)
        if self.golden_checkpoint is None and self.save_golden:
            golden_path = self.checkpoint_dir / 'checkpoint_golden'
            self._save_checkpoint_data(golden_path, checkpoint_data, epoch, metric)
            self.golden_checkpoint = golden_path
            logging.info(f"Saved golden checkpoint: {golden_path}")

        # Regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}'
        self._save_checkpoint_data(checkpoint_path, checkpoint_data, epoch, metric)

        # Add to best checkpoints list
        record = CheckpointRecord(
            epoch=epoch,
            metric=metric,
            path=checkpoint_path,
            timestamp=time.time()
        )
        self.best_checkpoints.append(record)

        # Sort by metric (descending) and keep only best N
        self.best_checkpoints.sort(key=lambda x: x.metric, reverse=True)

        # TASK_881 scope item 5: this is the 4th checkpoint-deletion site in this codebase
        # (alongside the offline tools/ckpt_gc.py, the in-run prune_run_dir_by_score above, and
        # flax's own keep=N in trainer.py). It is DELIBERATELY left OUTSIDE the manifest
        # protection model rather than wired into it, for three reasons: (1) `--safe-checkpoints`
        # is default OFF (mcts_trader/training/cli.py) — this code path does not run unless an
        # operator opts in; (2) it is already self-bounded by `keep_best_n` (default 3), so it
        # cannot grow without bound the way the pre-881 flax `keep=1_000_000` + `overwrite=True`
        # combination could; (3) `SafeCheckpointManager` is a generic, reusable class with no
        # knowledge of `experiments/`, run naming, or manifest conventions — wiring it into the
        # protection model would mean either threading a `protect` set through its constructor
        # (a behavioural change to every existing caller of this class) or having it dynamically
        # load `tools/ckpt_gc.py` the way `safe_checkpoint.prune_run_dir_by_score` does, which
        # would fail closed on every construction if the policy module were ever unavailable —
        # a much larger blast radius than the bounded `keep_best_n` risk being documented here.
        # If this path is ever wired into a longer, higher-stakes run than it serves today, it
        # should call `resolve_coverage`/`_is_protected` from `tools/ckpt_gc.py` exactly as
        # `prune_run_dir_by_score` does before this loop, not reinvent a second policy.
        while len(self.best_checkpoints) > self.keep_best_n:
            old_record = self.best_checkpoints.pop()
            if old_record.path.exists() and old_record.path != checkpoint_path:
                if old_record.path != self.golden_checkpoint:
                    shutil.rmtree(old_record.path)
                    logging.debug(f"Removed old checkpoint: {old_record.path}")

        self._save_state()

        logging.info(f"Saved checkpoint: {checkpoint_path} (metric: {metric:.3f})")
        return checkpoint_path

    def _save_checkpoint_data(
        self,
        path: Path,
        data: Dict[str, Any],
        epoch: int,
        metric: float
    ):
        """Save checkpoint data using orbax or fallback to pickle."""
        path.mkdir(parents=True, exist_ok=True)

        try:
            # Try orbax first (preferred for JAX)
            import orbax.checkpoint as ocp
            checkpointer = ocp.StandardCheckpointer()
            checkpointer.save(path / 'model', data)
        except ImportError:
            # Fallback to pickle
            import pickle
            with open(path / 'checkpoint.pkl', 'wb') as f:
                pickle.dump(data, f)

        # Save metadata
        metadata = {
            'epoch': epoch,
            'metric': metric,
            'metric_history': self.metric_history[-50:],
        }
        with open(path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    def load_best_checkpoint(self) -> Tuple[Dict[str, Any], int]:
        """
        Load the best available checkpoint for rollback.

        Returns:
            Tuple of (checkpoint_data, epoch)
        """
        # Try best checkpoints first
        for record in self.best_checkpoints:
            if record.path.exists():
                data = self._load_checkpoint_data(record.path)
                if data is not None:
                    self.rollback_count += 1
                    self._save_state()

                    logging.info(f"Loaded checkpoint: {record.path} (metric: {record.metric:.3f})")
                    logging.info(f"Total rollbacks: {self.rollback_count}")

                    return data, record.epoch

        # Fall back to golden checkpoint
        if self.golden_checkpoint and self.golden_checkpoint.exists():
            data = self._load_checkpoint_data(self.golden_checkpoint)
            if data is not None:
                self.rollback_count += 1
                self._save_state()

                logging.warning("Rolled back to GOLDEN checkpoint (last resort)")
                return data, 0

        raise RuntimeError("No valid checkpoint found for rollback!")

    def _load_checkpoint_data(self, path: Path) -> Optional[Dict[str, Any]]:
        """Load checkpoint data."""
        try:
            # Try orbax first
            import orbax.checkpoint as ocp
            checkpointer = ocp.StandardCheckpointer()
            return checkpointer.restore(path / 'model')
        except (ImportError, Exception):
            # Fallback to pickle
            pkl_path = path / 'checkpoint.pkl'
            if pkl_path.exists():
                import pickle
                with open(pkl_path, 'rb') as f:
                    return pickle.load(f)

        logging.warning(f"Failed to load checkpoint: {path}")
        return None

    def reset_for_new_level(self):
        """Reset manager state for a new curriculum level."""
        self.best_metric = float('-inf')
        self.metric_history = []
        self.epochs_since_improvement = 0
        self.level_start_metric = None
        # Keep rollback count for tracking across levels
        # Keep golden checkpoint as ultimate fallback

        logging.info("SafeCheckpointManager reset for new level")

    def get_stats(self) -> Dict[str, Any]:
        """Get current manager statistics."""
        return {
            'best_metric': self.best_metric,
            'rolling_avg': self.get_rolling_average(),
            'epochs_since_improvement': self.epochs_since_improvement,
            'rollback_count': self.rollback_count,
            'num_checkpoints': len(self.best_checkpoints),
            'has_golden': self.golden_checkpoint is not None,
            'history_length': len(self.metric_history),
        }


# Convenience function for typical usage
def create_level_checkpoint_manager(
    base_dir: str,
    level: int,
    difficulty: str = 'medium'
) -> SafeCheckpointManager:
    """
    Create a SafeCheckpointManager with level-appropriate settings.

    Args:
        base_dir: Base checkpoint directory
        level: Curriculum level (0-14)
        difficulty: 'easy', 'medium', 'hard', or 'expert'
    """
    # Settings based on difficulty
    settings = {
        'easy': {
            'patience': 100,
            'rollback_threshold': 1.0,
            'keep_best_n': 2,
            'rolling_window': 30,
        },
        'medium': {
            'patience': 50,
            'rollback_threshold': 0.5,
            'keep_best_n': 3,
            'rolling_window': 20,
        },
        'hard': {
            'patience': 30,
            'rollback_threshold': 0.3,
            'keep_best_n': 5,
            'rolling_window': 15,
        },
        'expert': {
            'patience': 20,
            'rollback_threshold': 0.2,
            'keep_best_n': 5,
            'rolling_window': 10,
        },
    }

    config = settings.get(difficulty, settings['medium'])

    return SafeCheckpointManager(
        checkpoint_dir=f"{base_dir}/level_{level}",
        **config
    )
