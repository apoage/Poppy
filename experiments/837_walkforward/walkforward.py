#!/usr/bin/env python3
"""TASK_837 / R4 — walk-forward split harness (the graduation instrument).

The definition-of-working says UNSEEN YEARS; before this module nothing in the repo could
produce an unseen-years number (training and 835-eval both run the same pinned 2021-2025
window). This harness BUILDS THE RAILS — it trains nothing:

  1. FOLD MATERIALIZATION — per-fold periods-files in the exact 821 format
     (``experiments/821_planner_daily/period_821_train.json``: ``periods[0]``
     {name, start_date, end_date, assets}; train_jax_fast.py:223-231 and 835's
     ``--periods-file`` both read periods[0] ONLY), plus per-fold POINT-IN-TIME universes,
     plus job manifests consumable by ``tools/deploy/run_queue.sh`` (whitespace format:
     ``<job_id> <gpu|cpu> <mem> <command...>``) and by
     ``experiments/835_stage4_eval/eval_vector.py`` (its CLI; imported by manifest, never
     modified).
  2. PIT UNIVERSE (load-bearing) — ``trader/data.py:401 build_pit_universe(db_path,
     min_bars)`` carries NO as-of date: its coverage floor counts ALL 1-min bars
     (``SELECT COUNT(*) FROM ohlcv WHERE asset_id=?`` — unbounded in time), so it CANNOT
     derive a universe as-of train-end. The PIT derivation is therefore implemented HERE
     (read-only queries; ZERO edits to trader/data.py) — rule pre-registered in SPEC.md:
     build_pit_universe's USD-crypto filter (imported ``STABLE_FIAT``) + a coverage floor +
     coverage ranking, all TIME-BOUNDED at train-end, counted on ``ohlcv_daily`` (LOUD
     DEVIATION, see SPEC §PIT: daily bars instead of 1-min bars — the canonical audited
     daily store, and fixture-daily == db-daily is LEDGER-official per 830/E0-REAL, so fold
     universes are derivable and leak-assertable without touching trading_data.db).
     The pinned-64 universe is emitted alongside, labeled LEAKY-UNIVERSE (821 continuity).
  3. FOLD CORRIDOR — per-eval-window floor/ceiling/ref via the 830 corridor machinery
     (``experiments/830_corridor/corridor.py`` loaded by importlib, the 832 pattern; never
     modified). The eval windows are shorter than the panel loader's ``min_active=365``
     liquidity filter tolerates (the 831-EXT crash precedent: windows <366d zero the
     universe), so the corridor panel is padded BACKWARD (past data only — never beyond
     eval-end) by a pre-registered ``--pad-days`` and every corridor number is computed on
     the EVAL-WINDOW DECISION SLICE only:
       floor_inenv = 830 ``replay_floor`` rets sliced to eval rows (830 row convention);
       reference   = ``trader.backtest.window_metrics`` (the gate-2 scorer: positions on
                     the whole panel, metrics ONLY on ts in [eval_start, eval_end));
       ceiling     = 830 ``oracle_positions`` scored by ``trader.backtest.regime_series``
                     on the same [r_lo, r_hi) slice (identical audited accounting).
  4. AGGREGATION — per-fold 835 scorecards + per-fold corridors -> ONE walk-forward
     summary JSON + a LEDGER-convention row. Per-fold corridor positions are ALWAYS shown;
     the pooled block exists only underneath the per-fold rows (never a pooled Sharpe
     without the per-fold view).

LEAK-FREE GATES AS CODE (the deliverable): ``validate`` parses the GENERATED artifacts
(it never trusts construction) and FAILS LOUD if any fold violates:
  ASSERT-1  no train/eval date overlap in any generated fold artifact (train + eval
            periods-files re-parsed with BOTH the training loader's UTC-midnight parse
            (vectorbt_env.py:660 ``timestamp <= end_ts`` inclusive semantics; TASK_869:
            was local-midnight) and the
            backtest UTC parse — the conservative envelope must be disjoint);
  ASSERT-2  no eval-window data can enter a train periods-file: the loader's own ingest
            query is run against the db with the PARSED train-file bounds and the max
            ingested timestamp (minute AND daily tables) must precede eval-start;
            plus PIT re-derivation equality (universe unchanged when derived from data
            <= train-end only — a tampered universe is REJECTED), manifest cross-checks
            (a train job may only reference a fold train periods-file; an eval job must
            reference the fold's eval_pit file IDENTICALLY — same-window substitutes like
            the LEAKY file are rejected), and a corridor panel-bound check (corridor
            never loaded data past eval-end);
  POLICY-PIN every parsed fold window and universe-rule parameter is compared against the
            PRE-REGISTERED constants (POLICY) — never against the artifacts' own
            (tamperable) rule blocks; internally-consistent tampers and materialize
            parameter typos are rejected. The fixture smoke proves the gates in BOTH
            directions: an unmutated control clone validates CLEAN and five corrupted
            clones are each rejected BY THE TARGETED gate (named in the rejection).

Pre-registered REAL folds (planner, 2026-07-08 — the harness implements, NEVER invents):
  fold1: train 2021-01-01..2023-12-31 -> eval 2024-01-01..2024-12-31
  fold2: train 2021-01-01..2024-12-31 -> eval 2025-01-01..2025-09-01
Pre-registered SMOKE folds (sized to the fixture minute slice 2025-09-13..2025-12-12,
SPEC §smoke): smokefold1 train 2025-09-14..2025-10-31 -> eval 2025-11-01..2025-11-30;
smokefold2 train 2025-09-14..2025-11-30 -> eval 2025-12-01..2025-12-11.

FACTORY / SCHEDULED split (830/831/835 house style): this module + the fixture smoke are
the factory deliverable. Fold TRAININGS are scheduled (phase-2+ planner decision, out of
scope); the scheduled re-materialization against ``trading_data.db`` (read-only) must
reproduce the emitted fold universes content-identically (identity contract in SPEC §PIT:
volatile fields ``generated``/``db``/``minute_coverage_check`` excluded; periods-files
byte-identical). CPU-only (``JAX_PLATFORMS=cpu``), fixture db only in this task,
read-only queries, no npz.
"""
import os
# CPU-only, deterministic; keep JAX off the GPU (factory/smoke job — house style).
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import contextlib
import errno
import hashlib
import importlib.util
import json
import shutil
import sqlite3
import stat
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone, date

import numpy as np

# Run from anywhere: repo root on the path so `trader` imports cleanly.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from trader import backtest, config as trader_config
from trader.config import STABLE_FIAT

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURE_DB = os.path.join(_REPO, "fixtures", "trading_fixture.db")
PINNED_PERIODS = os.path.join(_REPO, "experiments", "821_planner_daily",
                              "period_821_train.json")
CORRIDOR_830 = os.path.join(_REPO, "experiments", "830_corridor", "corridor.py")

# ---- pre-registered fold policy (planner 2026-07-08; TASK_837 spec verbatim) ----
REAL_FOLDS = (
    {"fold_id": "fold1", "train": ("2021-01-01", "2023-12-31"),
     "eval": ("2024-01-01", "2024-12-31")},
    {"fold_id": "fold2", "train": ("2021-01-01", "2024-12-31"),
     "eval": ("2025-01-01", "2025-09-01")},
)
# ---- pre-registered smoke folds (SPEC §smoke — sized to fixture minute coverage) ----
SMOKE_FOLDS = (
    {"fold_id": "smokefold1", "train": ("2025-09-14", "2025-10-31"),
     "eval": ("2025-11-01", "2025-11-30")},
    {"fold_id": "smokefold2", "train": ("2025-09-14", "2025-11-30"),
     "eval": ("2025-12-01", "2025-12-11")},
)

# PIT qualification rule (pre-registered, SPEC §PIT): USD-quoted, non-stable/fiat
# (build_pit_universe's filter, STABLE_FIAT imported), >= MIN_DAYS daily bars with
# timestamp <= as-of (train-end UTC midnight), ranked by that count DESC (tie: asset_id
# ASC), capped at N_ASSETS. TWO deviations from build_pit_universe, both LOUD in SPEC.md:
# daily-bar counts instead of its 1-min counts, and the coverage RANKING + cap are NEW
# policy (build_pit_universe has no ranking/cap — it returns ALL qualifying, asset_id-
# sorted; the ranking mirrors the training loader's own coverage ordering instead).
PIT_MIN_DAYS = 180
PIT_N_ASSETS = 64
SMOKE_PIT_N_ASSETS = 8

# Pre-registration PIN (verify-panel MAJOR-1 fix, 2026-07-09): ``validate`` re-checks
# every generated artifact against THESE constants — never against the (tamperable)
# artifact's own rule block or fold_manifest alone. An internally-consistent tamper
# (windows or universe rule edited coherently across all files) and an honest
# ``materialize --min-days 30`` typo are both REJECTED. Only pre-registered fold sets
# validate (the harness implements splits, it never invents them).
POLICY = {
    "real": {"folds": {f["fold_id"]: f for f in REAL_FOLDS},
             "n_assets": PIT_N_ASSETS, "min_days": PIT_MIN_DAYS},
    "smoke": {"folds": {f["fold_id"]: f for f in SMOKE_FOLDS},
              "n_assets": SMOKE_PIT_N_ASSETS, "min_days": PIT_MIN_DAYS},
}

# Corridor conventions (830 lineage).
PAD_DAYS = 366          # backward pad: covers strategy warmup AND the min_active=365 filter
GATE_COSTS = (0, 21)    # bp/side cost tiers — gross + net@21 (fee16+slip5), the 830 SPEC tiers
MIN_EVAL_DECISIONS = 6  # regime_series' own floor; fewer eval decisions = fail loud

# Training arm mirrored VERBATIM from experiments/828_stage3_train/jobs_phase1.manifest
# (the preflight-revised phase-1: cost0-first; fee16 is a CONTINUATION decision, out of
# scope here). The eval manifest carries the SAME operator (hold_bias 0.0 — 835's CLI
# default 3.0 is the jobs.manifest value, wrong for this arm; explicit beats default).
ARM = {
    "name": "cost0",
    "train_flags": ("--jax-env --db-path {db} --periods-file {pf} --bar-minutes 1440 "
                    "--vector-action --vector-proposal-scheme dithered-book --vector-k 16 "
                    "--vector-hold-bias 0.0 --vector-dither-scale 1.0 --reward-hybrid "
                    "--reward-attributable --model-size small --sims 32 --batch-size 256 "
                    "--parallel-envs 4 --steps 256 --fee 0 --slippage 0 --epochs 2000 "
                    "--seed {seed} --snapshot-every 100 --name {name}"),
    "eval_flags": ("--db {db} --periods-file {pf} --bar-minutes 1440 --n-assets {n_assets} "
                   "--vector-proposal-scheme dithered-book --vector-k 16 "
                   "--vector-hold-bias 0.0 --vector-dither-scale 1.0 --sims 32 "
                   "--checkpoint {ckpt} --eval-seed 0 --tag {tag} --out-dir {out_dir}"),
}
SEEDS = (0, 1, 2)
HOLD_COLLAPSE_PCT = 99.0   # the 821 signature (835 convention)


class LeakError(RuntimeError):
    """A generated fold artifact violates the leak-free gates. ALWAYS fatal."""


class MaterializeRefusal(RuntimeError):
    """TASK_904A: the module's WRITE-REFUSAL class, raised by ``materialize`` when it
    would write into a directory whose content it did not generate, or whose content it
    would silently downgrade. ALWAYS fatal — never a silent skip."""


# TASK_904A (cycle 2) — WHY THE GATE IS A FLAG AND NOT A SENTINEL.
# Cycle 1 gated overwrites on a sentinel STRING written into the very file it guarded
# ("_materializer": "TASK_837/walkforward.materialize"). That failed in BOTH directions,
# and both were demonstrated:
#   * forgeable — the constant is public and bound to nothing (not to content, not to the
#     db, not to the fold set), so pasting it into any hand-made directory bought write
#     access to exactly the tree the guard existed to protect;
#   * and it permanently LOCKED OUT the SPEC.md-scheduled real-db re-materialization,
#     because the tracked folds/real/ tree was generated 2026-07-08 and can never carry a
#     sentinel invented afterwards.
# A presence-based credential written into the file it guards cannot be both unforgeable
# and grandfatherable. So:
#   * the GATE is the operator's intent — an explicit opt-in (``allow_overwrite``,
#     surfaced as ``--overwrite-existing``). Nothing inside a directory can grant itself
#     write permission any more.
#   * PROVENANCE is DISCLOSURE, never a lock: the manifest records what produced the tree
#     (db, db class, fold set, content hash) so a re-materialization can say out loud what
#     it is replacing and flag disagreements. It grants nothing — with the refusals
#     below, which are the only things it can deny.
#   * refusal 1: the IDENTITY FENCE (TASK_904A blocker fix, see
#     ``_authorize_canonical_write`` / ``_guard_write_path`` below) — the canonical
#     tracked real-fold tree is protected by WHAT IT IS (the git checkout the DESTINATION
#     itself lives in, matched by path and by ``(st_dev, st_ino)``), never by what its own
#     manifest claims about its provenance and never by where this script happens to sit.
#   * refusal 2: FIXTURE-over-REAL, flag or no flag, for every OTHER real-provenance
#     tree. Overwriting a real-db fold tree with fixture-derived content is never what an
#     operator means.
#   * refusal 3 (cycle 3 / F1): a provenance block that DISAGREES WITH ITSELF. Refusal 2
#     is computed from the db class, so the class must never be READ OUT of the manifest
#     as a string — a directory whose provenance says {"universe_db": "trading_data.db",
#     "db_kind": "fixture"} would otherwise talk its way past the one refusal no flag
#     overrides, by lying in the very field the refusal reads (and the diagnostic would
#     echo the lie back as "fixture -> fixture"). The class is therefore ALWAYS re-derived
#     from ``universe_db`` via ``_db_kind``; a stored ``db_kind`` that disagrees with the
#     re-derivation is a corruption signal and is refused with both values named.
_PROVENANCE_KEY = "_provenance"
_MATERIALIZER_ID = "TASK_837/walkforward.materialize"
_OVERWRITE_FLAG = "--overwrite-existing"

# =============================================================================
# TASK_904A BLOCKER FIX (run-42 escalation) — DESTINATION-FIRST IDENTITY
# =============================================================================
# The first cut of the fence was ``os.path.realpath(out_root) == _CANONICAL_REAL_ROOT``,
# where that constant was derived ``__file__``-relative (``EXP_DIR/folds/real``). It failed
# in two ways with ONE root cause: **the fence held the CALLER'S WORD about the
# destination, anchored to an accident of invocation, instead of the DESTINATION'S OWN
# IDENTITY.**
#
#   BLOCKER 1 — the anchor was ``__file__``-relative. Run the IDENTICAL unmodified script
#     from a COPY of this directory (a ``/tmp`` reconstruction of the tree, a HEAD-control
#     worktree A/B rig — both standing techniques in this repo) and point ``--out-root`` at
#     the real tree by absolute path: the anchor then names the COPY's ``folds/real``, the
#     string comparison fails, and the fence goes INERT. No refusal, no warning, write
#     proceeds. An inert fence is worse than no fence, because the record claims one exists.
#   BLOCKER 2 — the predicate required the caller to SELF-REPORT the destination's nature
#     (``fold_set == "real" and this_kind == "fixture"``), so a caller who merely
#     DISAGREED with the destination switched the guard off:
#     ``materialize --fold-set smoke --out-root <canonical folds/real> --overwrite-existing``
#     uses the perfectly legal value ``smoke``, fails that first conjunct, ALSO fails the
#     plain fixture-over-real check (whose ``prior_kind`` is re-derived from the canonical
#     tree's own manifest, which self-reports ``"fixture"``), and writes 8-asset smoke-window
#     content into the tracked 64-asset real rails.
#
# The inversion, in three moves:
#
#  (1) ANCHOR = THE DESTINATION'S OWN REPOSITORY. Walk UP from ``realpath(dest)`` to the
#      nearest ancestor holding a ``.git`` entry; the canonical tree is that root's
#      ``experiments/837_walkforward/folds/real``. Invocation-independent by construction —
#      it does not matter which copy of this script runs, or from where, because the
#      question asked is about the DESTINATION.
#      CLAIM CORRECTION (cycle 3): as first shipped this sentence was true of THIS anchor
#      and false of the module, because a SECOND anchor survived elsewhere — ``FIXTURE_DB``,
#      still ``__file__``-relative, still feeding the db classifier that the unlock's
#      "real-classed --db" conjunct consulted. Run from a worktree, an absolute path to the
#      real project fixture matched no fixture spelling the copy knew about and classified
#      "real"; the unlock completed and fixture content overwrote a real-labelled canonical
#      tree at exit 0. Invocation-independence is a property that has to hold for EVERY
#      anchor in the guard, not just the one being written about. Both are now
#      destination-anchored (see the polarity note above ``_db_class``).
#      A ``/tmp`` scratch copy outside any checkout
#      resolves to "not a tracked tree" (correct — that is what scratch copies are for),
#      while the real tree named by absolute path from anywhere resolves to itself. The
#      script's own repo is retained ONLY as an ADDITIONAL candidate (a union of candidates
#      can only ever refuse MORE), never as the primary. Identity is compared by path AND by
#      ``(st_dev, st_ino)``, so a bind-mount or a second mount spelling of the same
#      directory cannot present itself as a different destination.
#      If identity cannot be resolved, the guard REFUSES LOUDLY — it never goes inert.
#  (2) PREDICATE = DESTINATION-ONLY, FAIL-CLOSED. If the destination IS that tree, EVERY
#      write is refused unless an affirmative, fully CONSISTENT unlock is present:
#      ``fold_set == "real"`` AND a POSITIVELY PROVEN real-classed ``--db`` AND the explicit
#      override flag. Disagreement with the destination is now DISQUALIFYING, where before
#      it was exculpatory.
#      CLAIM CORRECTION (cycle 3): "verifiably real-classed" was the load-bearing word and
#      it was not implemented. The conjunct asked a classifier whose fallback for "cannot
#      tell" was ``"real"``, so the conjunct was satisfied by a FAILURE to classify — the
#      opposite of verification. It now asks ``_require_positively_real_db``, where
#      unprovable is a refusal. The scope of the resulting guarantee, stated exactly: the
#      project fixture cannot author this tree's content at any path or spelling, nor can a
#      renamed byte-identical copy of it, nor can any db whose relation to the fixture
#      cannot be established. A db that is NOT byte-identical to the pinned fixture is
#      treated as real, so fixture-DERIVED-then-modified content is still authorizable —
#      that is a deliberate limit (it is what makes the test suite's stand-in dbs and any
#      legitimate real-db copy usable), not an oversight.
#      TRUST DEPENDENCY, stated because the guarantee above is contingent on it (found by
#      the cycle-3 re-verify): the "renamed byte-identical copy" half of that guarantee
#      rests ENTIRELY on the correctness of the tracked pin ``fixtures/FIXTURE_IDENTITY.json``.
#      If that pin's sha256 is stale, hand-edited or corrupt, a renamed byte-identical copy
#      of the real fixture classifies as ``_DB_REAL`` and IS authorized. It bites only where
#      the db path does not carry the fixture's basename AND no local fixture file is
#      available for inode comparison at any candidate root -- i.e. exactly the
#      worktree/reconstruction case the pin leg exists to cover. The bare-default
#      ``--db fixtures/trading_fixture.db`` path is unaffected (spelling evidence wins there).
#      Note also that this module does NOT call ``fixtures/fixture_identity.py``'s
#      ``verify_or_die``, unlike several other experiment smoke scripts, so nothing here
#      cross-checks the pin against a live fixture file before trusting it. Tracked as a
#      named remainder on TASK_904A rather than fixed in cycle 3.
#  (3) BINDING = THE SHARED WRITE CHOKE POINT (``_guard_write_path``, called by
#      ``_write_json``), not only ``materialize``. A fence on one entry point is not a
#      fence: bare ``corridor`` rewrote ``folds/real/fold*/corridor.json`` with no gate
#      whatsoever and moved the tracked tree hash twice in two days. The binding is
#      ADDITIVE and REFUSE-ONLY — it authorizes nothing that was not already going to
#      happen and it restructures none of the corridor/aggregate writers (that two-phase
#      redesign is TASK_904B and stays out of scope).
_CANONICAL_REAL_REL = os.path.join("experiments", "837_walkforward", "folds", "real")

# Stack of write scopes currently authorized for the canonical tree (realpaths). Empty by
# default: outside an explicitly authorized scope, EVERY write into that tree is refused.
# Single-threaded by design (this module is a CLI harness); a threaded caller would need
# this to be thread-local.
_WRITE_UNLOCK = []


def _repo_root_of(path):
    """The nearest ancestor of ``path`` that holds a ``.git`` entry — a DIRECTORY for an
    ordinary checkout, a FILE for a git worktree or submodule, both counted — or None if
    ``path`` is not inside a checkout at all. Walks the REALPATH, so a symlinked
    destination cannot hide its true location. ``path`` need not exist."""
    cur = os.path.realpath(path)
    while True:
        if os.path.exists(os.path.join(cur, ".git")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def _same_inode(a, b):
    """``(st_dev, st_ino)`` identity: True only when both exist and ARE the same filesystem
    entry (directory or file). Catches the spellings string comparison misses — bind
    mounts, a second mount of the same filesystem, a hardlinked path — in one move."""
    try:
        sa, sb = os.stat(a), os.stat(b)
    except OSError:
        return False
    return (sa.st_dev, sa.st_ino) == (sb.st_dev, sb.st_ino)


def _canonical_real_tree_hit(dest):
    """DESTINATION-FIRST identity. Is ``dest`` the canonical tracked real-fold tree of SOME
    git checkout — or a path inside it? Returns that canonical root's realpath on a hit,
    else None. ``dest`` may be a directory or a file inside one, and need not exist.

    NEVER GOES INERT: anything unexpected while resolving raises ``MaterializeRefusal``
    rather than quietly answering "not canonical". A guard that cannot resolve its own
    anchor must refuse — that was BLOCKER 1's real severity."""
    try:
        d = os.path.realpath(dest)
        candidates = []
        root = _repo_root_of(d)
        if root is not None:
            candidates.append(os.path.join(root, _CANONICAL_REAL_REL))
        # SECONDARY, additive only — never the primary anchor (script-relative derivation
        # is exactly what BLOCKER 1 broke). Kept so that a destination which somehow
        # resolves to no checkout is still fenced against THIS repo's canonical tree.
        candidates.append(os.path.join(_REPO, _CANONICAL_REAL_REL))
        for cpath in candidates:
            c = os.path.realpath(cpath)
            if d == c or d.startswith(c + os.sep) or _same_inode(d, c):
                return c
        return None
    except MaterializeRefusal:
        raise
    except Exception as e:      # noqa: BLE001 — fail-closed is the whole point
        raise MaterializeRefusal(
            f"REFUSING: cannot establish whether {dest!r} is the canonical tracked "
            f"real-fold tree ({type(e).__name__}: {e}). A guard that cannot resolve its "
            f"own anchor refuses rather than going inert. Nothing was written.")


def _assert_db_usable(db_path, who):
    """MAJOR fix — a falsy or unopenable ``--db`` must never make the write refusals INERT.

    The single two-valued ``_db_kind`` used to return a THIRD value (``"unknown"``) for a
    falsy path while both refusal branches gated on ``== "fixture"``, so ``--db ""`` (or a
    script interpolating an unset variable) switched them both off. The failure then
    surfaced as an uncaught
    ``sqlite3.OperationalError`` — or, for ``None``, a ``TypeError`` raised inside the
    guard's OWN diagnostic print — neither carrying the parseable ``MATERIALIZE REFUSED:``
    prefix an orchestrator reads. THIS call's db is therefore required to be an existing
    regular file, checked before anything is classified and before anything is written.

    Recorded PROVENANCE strings are a different matter: those legitimately name paths from
    other machines, so ``_db_kind_protective`` still tolerates them (and classifies an
    unresolvable one ``"real"`` — the PROTECTIVE direction, which is safe there and only
    there; the authorizing side uses ``_require_positively_real_db``, where an unresolvable
    db is a refusal)."""
    if not db_path:
        raise MaterializeRefusal(
            f"{who}: REFUSING — no db given ({db_path!r}). The fixture-over-real and "
            f"canonical-tree refusals are computed from THIS call's db class, so an "
            f"empty/None db would silently disable them (that is how the fence went "
            f"inert). Pass an explicit --db. Nothing was written.")
    if _resolve_db_file(db_path) is None:
        raise MaterializeRefusal(
            f"{who}: REFUSING — db {db_path!r} is not an existing regular file, so this "
            f"call's db class cannot be established and the write refusals that depend on "
            f"it would be inert. Nothing was written.")


def _authorize_canonical_write(dest, db_path, fold_set, allow_overwrite, who):
    """The DESTINATION-FIRST unlock (move 2 above). Returns the canonical root when
    ``dest`` IS the canonical tracked real-fold tree **and** the unlock is affirmatively
    complete; returns None when ``dest`` is simply not that tree; raises
    ``MaterializeRefusal`` otherwise. Pure — consults nothing inside the destination, and
    writes nothing.

    The unlock is a CONJUNCTION and every conjunct must AGREE with the destination:
      * ``fold_set == "real"`` — this tree holds the pre-registered REAL folds (64 assets,
        2021-2025 windows). ``smoke`` is a legal value that simply does not describe this
        destination; under the old caller-reported predicate that very DISAGREEMENT is what
        let 8-asset smoke content overwrite it.
      * a POSITIVELY PROVEN real-classed db (``_require_positively_real_db``, the
        AUTHORIZING polarity). The project fixture — which is the CLI default — can never
        author this tree's content, and neither can a renamed byte-identical copy of it,
        nor a db whose relation to the fixture cannot be established at all. Unknown is a
        refusal here, never a permit; that inversion is the cycle-3 BLOCKER fix.
      * the explicit override flag (``--overwrite-existing`` / ``allow_overwrite=True``).
    Anything less is refused, with the failing conjuncts named."""
    canonical = _canonical_real_tree_hit(dest)
    if canonical is None:
        return None
    _assert_db_usable(db_path, who)          # no third db class inside the fence
    missing = []
    if fold_set != "real":
        missing.append(
            f"--fold-set is {fold_set!r}, not 'real' — this tree holds the pre-registered "
            f"REAL folds, so a non-real fold set does not describe it and writing one here "
            f"would replace 64-asset real rails with a different universe and window")
    # AUTHORIZING POLARITY — positive proof required; "cannot classify" counts as "no".
    # This is the conjunct the cycle-3 BLOCKER lived in: it used to ask the PROTECTIVE
    # classifier, whose "cannot classify -> real" fallback GRANTED the write.
    why_not_real = _require_positively_real_db(db_path, dest, who)
    if why_not_real is not None:
        missing.append(
            f"a POSITIVELY PROVEN real-classed --db (the CLI default is the project "
            f"fixture; only a db proven NOT to be it may author this tree's content, and "
            f"'cannot prove' counts as 'no') — {why_not_real}")
    if not allow_overwrite:
        missing.append(f"{_OVERWRITE_FLAG} (CLI) / allow_overwrite=True (API) was not given")
    if missing:
        raise MaterializeRefusal(
            f"{who}: REFUSING to write into {_rel(canonical)} — this IS the canonical "
            f"tracked real-fold tree, identified by the DESTINATION's own identity (the git "
            f"checkout it lives in, matched by path and by (st_dev,st_ino)) — NOT by where "
            f"this script sits, and NOT by what the directory's own manifest claims about "
            f"its provenance (that manifest records universe_db='fixtures/"
            f"trading_fixture.db' because the tree really was built from the fixture in "
            f"July, so its self-description can never supply this refusal). Every write "
            f"here requires an affirmative, fully consistent unlock; MISSING: "
            + "; ".join(missing)
            + ". Point the output at a scratch directory instead. Nothing was written.")
    return canonical


@contextlib.contextmanager
def _canonical_write_scope(dest, db_path, fold_set, allow_overwrite, who):
    """Wrap a writer's writes in this. It re-runs ``_authorize_canonical_write`` (pure, so
    running it twice is free) and, when the destination IS the canonical tree with a
    complete unlock, opens the choke point for THAT subtree only, for the duration only.
    Outside such a scope ``_guard_write_path`` refuses every write into that tree."""
    root = _authorize_canonical_write(dest, db_path, fold_set, allow_overwrite, who)
    if root is None:
        yield
        return
    _WRITE_UNLOCK.append(root)
    try:
        yield
    finally:
        _WRITE_UNLOCK.pop()


def _guard_write_path(path):
    """THE SHARED WRITE CHOKE POINT (move 3 above). Every write in this module passes
    through here (``_write_json``, plus the two job-manifest writes that use bare
    ``open``), so the destination-identity refusal covers ``corridor`` and ``aggregate``
    too — not only ``materialize``. Bare ``corridor`` rewriting
    ``folds/real/fold*/corridor.json`` with NO gate at all is what moved the tracked tree
    hash twice in two days, and "a fence on one entry point is not a fence".

    Refuse-only: it can never authorize a write, and it restructures none of the writers
    (their two-phase redesign is TASK_904B). Returns ``path`` unchanged on success."""
    canonical = _canonical_real_tree_hit(path)
    if canonical is None:
        return path                     # not the protected tree — nothing to gate
    p = os.path.realpath(path)
    for root in _WRITE_UNLOCK:
        if p == root or p.startswith(root + os.sep):
            return path                 # an affirmatively authorized scope is open
    raise MaterializeRefusal(
        f"REFUSING to write {_rel(path)} — it is inside {_rel(canonical)}, the canonical "
        f"tracked real-fold tree (identified by the destination's own git checkout, by "
        f"path and by (st_dev,st_ino)), and no authorized write scope is open for it. "
        f"Writes into this tree come ONLY from a writer that affirmatively unlocked it: "
        f"--fold-set real AND a real-classed --db AND {_OVERWRITE_FLAG}. For `materialize` "
        f"/ `corridor`, pass all three. For `aggregate`, point --out at a run directory "
        f"(e.g. runs/<id>/WF_SUMMARY.json): the tracked fold tree is pre-registered RAILS, "
        f"not a results directory. Nothing was written.")


# =============================================================================
# TASK_904A CYCLE-3 BLOCKER FIX — ONE CLASSIFIER, TWO OPPOSITE POLARITIES, SPLIT APART
# =============================================================================
# The cycle-2 cut left a SINGLE two-valued ``_db_kind`` serving two consumers whose
# polarity is OPPOSITE, with a "cannot classify -> real" fallback:
#
#   * ``_prior_db_kind`` (PROTECTIVE) asks "which class should I ASSUME so the directory
#     gets protected HARDER?". There, "unknown -> real" is exactly right.
#   * ``_authorize_canonical_write`` (AUTHORIZING) asks "may this caller WRITE the
#     canonical tracked tree?" and required ``_db_kind(db) == "real"``. There the SAME
#     value means "you are authorized" — so the conservative fallback GRANTED the write.
#
# Compounding it, fixture identity was ``__file__``-anchored (``FIXTURE_DB`` derived from
# this script's ``_REPO``). Run the script from a worktree or a /tmp reconstruction — where
# the GITIGNORED ``fixtures/trading_fixture.db`` does not exist — and an ABSOLUTE path to
# the actual project fixture matched neither ``samefile`` nor the ``_rel`` spelling, fell
# through the fallback, and classified ``"real"``. Live result: fixture content overwrote a
# real-labelled canonical tree at exit 0, while the guard's own diagnostic printed
# "(real -> real)" about the project fixture. Same hole for ``corridor``.
#
# The fix has three parts, and the first is the load-bearing one:
#
#  (1) AUTHORIZATION IS NEVER GRANTED BY A FALLBACK, A DEFAULT, OR A FAILURE TO CLASSIFY.
#      The classification is a TRI-STATE (``_DB_FIXTURE`` / ``_DB_REAL`` / ``_DB_UNPROVEN``)
#      and the two polarities are SEPARATE, differently-named functions over it:
#      ``_db_kind_protective`` (collapses UNPROVEN -> "real": protect harder) and
#      ``_require_positively_real_db`` (collapses UNPROVEN -> REFUSE: unknown is not a
#      permit). Neither can be mistaken for the other at a call site, and there is no
#      single boolean-ish classifier left for a future reader to point the wrong way.
#  (2) FIXTURE IDENTITY IS DESTINATION-ANCHORED, NEVER ``__file__``-ANCHORED. The roots
#      against which "the project fixture" is resolved come from the DESTINATION's own repo
#      (``_repo_root_of``, as the canonical-tree fence already does), with this script's
#      ``_REPO`` retained only as an ADDITIONAL candidate (more candidates can only refuse
#      more). ``FIXTURE_DB`` remains the CLI DEFAULT VALUE and is never again an identity
#      oracle.
#  (3) IF FIXTURE IDENTITY CANNOT BE ESTABLISHED, THAT IS "NOT POSITIVELY REAL" -> REFUSE.
#      Identity is established by either evidence source: the fixture FILE itself under a
#      candidate root (``(st_dev, st_ino)`` identity), or its TRACKED content pin
#      ``fixtures/FIXTURE_IDENTITY.json`` (TASK_873: sha256 + size_bytes). The pin is
#      TRACKED, so it survives into exactly the worktree/reconstruction case where the
#      gitignored fixture does not — which is how the reported reproduction is caught by
#      CONTENT rather than by path. With neither source available, nothing can be proven
#      and the authorizing consumer refuses.
#
# The change is REFUSE-ONLY: every input the cycle-2 code refused is still refused, and the
# only movement is UNPROVEN inputs flipping from "authorized" to "refused".
_DB_FIXTURE = "fixture"
_DB_REAL = "real"
_DB_UNPROVEN = "unproven"

_FIXTURE_REL = os.path.join("fixtures", "trading_fixture.db")
_FIXTURE_PIN_REL = os.path.join("fixtures", "FIXTURE_IDENTITY.json")
_FIXTURE_BASENAME = os.path.basename(_FIXTURE_REL)

_SHA256_CACHE = {}


def _sha256_file(path):
    """Content digest, memoized on ``(realpath, st_dev, st_ino, st_size, st_mtime_ns)``.
    Only ever reached for a db whose SIZE already equals the pinned fixture size, so the
    24GB ``trading_data.db`` is never hashed. Returns None if it cannot be read."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (os.path.realpath(path), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
    if key in _SHA256_CACHE:
        return _SHA256_CACHE[key]
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    _SHA256_CACHE[key] = h.hexdigest()
    return _SHA256_CACHE[key]


def _read_fixture_pin(pin_path):
    """The TRACKED fixture-identity pin (``fixtures/FIXTURE_IDENTITY.json``, TASK_873) read
    as a CONTENT oracle: ``{"sha256", "size_bytes"}`` or None.

    Why it matters here: the pin is TRACKED while the fixture itself is GITIGNORED, so the
    pin is present in a git worktree / a ``/tmp`` reconstruction where the fixture file is
    absent — precisely the case whose missing ``samefile`` target let an absolute path to
    the real project fixture classify as ``"real"``.

    None means "no evidence from this root", never "assume fine": with no evidence from any
    root the class is ``_DB_UNPROVEN`` and the authorizing consumer refuses. Read through
    the shared guarded reader, so a FIFO/symlink/garbage pin cannot hang or crash us."""
    try:
        obj = _load_json_regular_file(pin_path, MaterializeRefusal,
                                      f"fixture identity pin {_rel(pin_path)}")
    except Exception:               # noqa: BLE001 — absent/unreadable/garbage == no evidence
        return None
    sha, size = obj.get("sha256"), obj.get("size_bytes")
    if isinstance(sha, str) and len(sha) == 64 and isinstance(size, int) and size > 0:
        return {"sha256": sha, "size_bytes": size}
    return None


def _resolve_db_file(db_path):
    """A ``--db`` / recorded-provenance path turned into an EXISTING regular file, or None.
    Deliberately the SAME resolution ``_assert_db_usable`` has always done (the path as
    given, then ``_REPO``-relative) so that nothing is newly ACCEPTED by this refactor."""
    if not db_path:
        return None
    for cand in (db_path, os.path.join(_REPO, db_path)):
        if os.path.isfile(cand):
            return cand
    return None


def _fixture_anchor_roots(dest):
    """The roots against which "the project fixture" is resolved, DESTINATION FIRST.

    ``dest=None`` means "no destination in question" (the protective polarity, which only
    asks whether a recorded string names the fixture) and yields this script's ``_REPO``
    alone. Otherwise the destination's own repo root leads and ``_REPO`` follows as an
    ADDITIONAL candidate — a union of candidates can only ever refuse MORE. If the
    destination's repository cannot be resolved at all the guard REFUSES rather than
    quietly dropping a candidate."""
    roots = []
    if dest is not None:
        try:
            r = _repo_root_of(dest)
        except Exception as e:      # noqa: BLE001 — fail-closed, never inert
            raise MaterializeRefusal(
                f"REFUSING: cannot resolve the repository of destination {dest!r} to "
                f"establish fixture identity ({type(e).__name__}: {e}). A guard that "
                f"cannot resolve its own anchor refuses rather than going inert. Nothing "
                f"was written.")
        if r is not None:
            roots.append(r)
    if _REPO not in roots:
        roots.append(_REPO)
    return roots


def _db_class(db_path, dest=None):
    """TRI-STATE, DESTINATION-ANCHORED db classification. Returns ``(class, why)`` where
    class is exactly one of ``_DB_FIXTURE`` / ``_DB_REAL`` / ``_DB_UNPROVEN`` and ``why``
    is the human sentence the refusing consumer quotes.

    Do NOT call this directly to make a decision — pick a POLARITY:
    ``_db_kind_protective`` or ``_require_positively_real_db``. That split is the whole
    point of the cycle-3 fix (see the design note above).

    Order of evidence, most portable first:
      1. SPELLING — a path whose basename is ``trading_fixture.db`` names the fixture, no
         filesystem needed. This is what classifies recorded provenance strings from other
         machines, and it alone already catches the reported reproduction (an absolute path
         to the project fixture). A real db that happens to carry that basename is
         mis-called "fixture", which is the refusing direction in BOTH consumers.
      2. INODE IDENTITY — ``(st_dev, st_ino)`` against ``<root>/fixtures/trading_fixture.db``
         for each candidate root: catches copies-by-symlink, bind mounts and second mount
         spellings.
      3. CONTENT IDENTITY — sha256 + size against each root's TRACKED pin: catches a
         RENAMED byte-identical copy of the fixture, and works where the gitignored fixture
         file is absent.
      4. ESTABLISHMENT — only if at least one of sources 2/3 was AVAILABLE may this return
         ``_DB_REAL``. Otherwise "not the fixture" is unproven, not true."""
    if not db_path:
        return _DB_UNPROVEN, (
            f"no db given ({db_path!r}), so nothing can be classified at all")

    # (1) SPELLING.
    if os.path.basename(os.path.normpath(str(db_path))) == _FIXTURE_BASENAME:
        return _DB_FIXTURE, (
            f"db {db_path!r} NAMES the project fixture ({_FIXTURE_REL}) — matched on the "
            f"path itself, so it holds no matter which checkout this script runs from")

    roots = _fixture_anchor_roots(dest)
    resolved = _resolve_db_file(db_path)
    if resolved is None:
        return _DB_UNPROVEN, (
            f"db {_rel(db_path)!r} is not an existing regular file here, so whether it is "
            f"the project fixture cannot be established")

    # (2) INODE IDENTITY against each candidate root's fixture file.
    fixture_file_seen = False
    for root in roots:
        cand = os.path.join(root, _FIXTURE_REL)
        if not os.path.isfile(cand):
            continue
        fixture_file_seen = True
        if _same_inode(resolved, cand):
            return _DB_FIXTURE, (
                f"db {_rel(db_path)!r} IS {os.path.join(_rel(root), _FIXTURE_REL)} by "
                f"(st_dev, st_ino) identity")

    # (3) CONTENT IDENTITY against each candidate root's TRACKED pin.
    pin_seen = False
    try:
        size = os.path.getsize(resolved)
    except OSError as e:
        return _DB_UNPROVEN, (
            f"db {_rel(db_path)!r} cannot be sized ({type(e).__name__}: {e}), so its "
            f"content cannot be compared against the pinned fixture identity")
    for root in roots:
        pin = _read_fixture_pin(os.path.join(root, _FIXTURE_PIN_REL))
        if pin is None:
            continue
        pin_seen = True
        if size != pin["size_bytes"]:
            continue                # cheap discriminator: never hashes the 24GB real db
        if _sha256_file(resolved) == pin["sha256"]:
            return _DB_FIXTURE, (
                f"db {_rel(db_path)!r} is BYTE-IDENTICAL to the project fixture pinned in "
                f"{os.path.join(_rel(root), _FIXTURE_PIN_REL)} (sha256 "
                f"{pin['sha256'][:12]}..., {size} bytes) — a renamed copy of the fixture "
                f"is still fixture content")

    # (4) ESTABLISHMENT — the cycle-3 hole, closed.
    if not (fixture_file_seen or pin_seen):
        return _DB_UNPROVEN, (
            f"fixture identity CANNOT BE ESTABLISHED for db {_rel(db_path)!r}: none of the "
            f"candidate roots {[_rel(r) for r in roots]} holds {_FIXTURE_REL} or its "
            f"tracked pin {_FIXTURE_PIN_REL}, so 'this db is not the project fixture' is "
            f"UNPROVEN. That unprovability is exactly what the __file__-anchored classifier "
            f"used to answer 'real'")
    return _DB_REAL, (
        f"db {_rel(db_path)!r} is proven NOT to be the project fixture (checked against "
        f"{[_rel(r) for r in roots]} by path, inode and pinned content)")


def _db_kind_protective(db_path):
    """PROTECTIVE POLARITY — "which class should I ASSUME so a directory gets protected
    HARDER?". Collapses the tri-state with ``_DB_UNPROVEN -> "real"``, because a
    real-provenance fold tree is the one worth protecting and an unclassifiable
    provenance string (a path from another machine) must not read as "merely fixture".
    Exactly two values, matching every consumer that gates on ``== "fixture"``.

    **NEVER call this to decide whether a caller MAY WRITE.** Here ``"real"`` means "treat
    it as precious"; in an authorizing consumer the same string means "you are permitted",
    and that collision is the cycle-3 BLOCKER. Authorizers call
    ``_require_positively_real_db`` — a different name, a different signature (it takes the
    DESTINATION), and a different collapse of the same tri-state."""
    cls, _why = _db_class(db_path, dest=None)
    return _DB_FIXTURE if cls == _DB_FIXTURE else _DB_REAL


def _require_positively_real_db(db_path, dest, who):
    """AUTHORIZING POLARITY — "is this db POSITIVELY PROVEN real-classed for THIS
    destination?". Returns None when it is, else the reason it is not (which the caller
    quotes in its refusal).

    ``_DB_UNPROVEN`` refuses exactly like ``_DB_FIXTURE``: authorization is never granted
    by a fallback, a default, or a failure to classify. Exhaustive over the tri-state BY
    CONSTRUCTION — a fourth class added later without a branch here raises instead of
    falling through to "authorized"."""
    cls, why = _db_class(db_path, dest=dest)
    if cls == _DB_REAL:
        return None
    if cls in (_DB_FIXTURE, _DB_UNPROVEN):
        return why
    raise MaterializeRefusal(
        f"{who}: REFUSING — unhandled db class {cls!r} for {db_path!r}. Authorization is "
        f"never granted by a fall-through. Nothing was written.")


def _load_json_regular_file(path, error, what, require=dict):
    """THE guarded JSON reader — ONE implementation, used by EVERY manifest reader in
    this module (materialize's provenance check, ``validate_folds``, ``aggregate``).
    Cycle 2 had the guard on one reader of three, so a FIFO at ``fold_manifest.json``
    still hung ``validate`` (which every ``materialize`` CLI call ends with) forever.

    Three hard rules, all of them fail-closed:

    1. FILE TYPE. A path that is not a regular file is refused before a single byte is
       read. Opening a FIFO blocks forever (this repo's own documented hang class —
       TASK_891/897) and a symlink points the read at content outside this tree. One
       ``S_ISREG`` check covers symlink, FIFO, socket, device and directory alike;
       deliberately NOT a taxonomy of exotic spellings (chasing those through an unbounded
       input space is TASK_891's recorded failure mode, and complexity inside a fail-closed
       guard is itself risk).
    2. NO TOCTOU (MINOR fix). The check is made on the FILE DESCRIPTOR that is then read,
       not on the path: ``os.open(..., O_NOFOLLOW | O_NONBLOCK)`` followed by ``os.fstat``
       of that same fd. The previous ``os.lstat(path)``-then-``open(path)`` pair left a
       window in which a substituted FIFO or symlink defeated the very check this function
       exists to make. ``O_NOFOLLOW`` refuses a symlinked path outright (ELOOP) and
       ``O_NONBLOCK`` means that even if a FIFO were somehow opened, the open itself cannot
       block — the ``fstat`` then refuses it before any read.
    3. TYPE OF THE PARSED VALUE (MAJOR fix). Every caller here reads a *manifest* and
       goes on to use it as a mapping. Valid JSON of the wrong shape (``[1,2,3]``,
       ``null``, ``"hello"``) used to sail through this reader and blow up downstream as a
       raw ``AttributeError``/``TypeError`` traceback — ``man.get(...)`` in
       ``validate_folds``, ``man["folds"]`` in ``aggregate`` — with NO parseable refusal
       prefix and a non-1 exit path, which falsified the claim that every guarded-reader
       refusal exits 1 through the handlers. Checked HERE, in the one place, rather than at
       each call site. ``require=None`` opts out for a caller that genuinely wants any
       JSON value.

    ``error`` is the caller's fatal class (LeakError / MaterializeRefusal) and ``what``
    the noun phrase naming the file in the caller's own voice. ABSENCE is not handled
    here — the ``os.open`` OSError propagates, because absence means different things to
    different callers (no provenance recorded / a leak-gate failure)."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as e:
        if e.errno == errno.ELOOP:
            # O_NOFOLLOW on a symlink. Same class of refusal as the S_ISREG check below,
            # and it says "regular file" for the same reason.
            raise error(
                f"{what} is a SYMLINK, not a regular file, so it is not opened at all: it "
                f"would point the read outside this tree. Remove/replace it deliberately.")
        raise                       # absence (ENOENT) and friends propagate, as documented
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise error(
                f"{what} is NOT a regular file (mode {stat.filemode(st.st_mode)}), so it "
                f"is not read at all: a FIFO there would block this process forever "
                f"(TASK_891/897 hang class), and a symlink would point the read outside "
                f"this tree. Remove/replace it deliberately.")
        try:
            # dup() so the `with` can own its own fd and the finally below stays simple.
            with os.fdopen(os.dup(fd), "r") as fh:
                obj = json.load(fh)
        except (OSError, ValueError) as e:
            # ValueError covers json.JSONDecodeError AND UnicodeDecodeError (binary or
            # non-UTF-8 content) — both take the clean refusal path, never a raw traceback.
            raise error(f"{what} cannot be read as JSON ({type(e).__name__}: {e}). "
                        f"Remove/replace it deliberately.")
    finally:
        os.close(fd)
    if require is not None and not isinstance(obj, require):
        raise error(
            f"{what} is valid JSON but the WRONG TYPE: expected a JSON object, got "
            f"{type(obj).__name__} ({json.dumps(obj)[:80]}). A manifest is a mapping; a "
            f"list/null/string there would otherwise surface downstream as a raw "
            f"AttributeError/TypeError traceback instead of this refusal. "
            f"Remove/replace it deliberately.")
    return obj


def _prior_db_kind(prov, where):
    """The db CLASS of a recorded provenance block, ALWAYS re-derived from its
    ``universe_db`` path — never read out of its stored ``db_kind`` string (cycle 3 / F1;
    see the design note above). A stored ``db_kind`` that DISAGREES with the
    re-derivation is treated as corruption and refused with both values named. Returns
    ``(prior_db, prior_kind)``, or ``(None, None)`` when the directory records NO
    ``universe_db`` at all — it makes no provenance claim, so there is nothing to
    re-derive and the class comparison in ``_assert_out_root_writable`` does not run. That
    is deliberately NOT "real": inventing a class for a directory that never recorded one
    would lock out every hand-made scratch out-root, and the tree actually worth protecting
    is protected by the DESTINATION-IDENTITY fence, not by this class comparison."""
    prior_db = (prov or {}).get("universe_db")
    stored = (prov or {}).get("db_kind")
    if not prior_db:
        if stored is not None:
            raise MaterializeRefusal(
                f"{where}: REFUSING — its recorded provenance DISAGREES WITH ITSELF: it "
                f"stores db_kind={stored!r} while recording NO universe_db to re-derive "
                f"that class from. The class is exactly what the fixture-over-real refusal "
                f"is computed from, so an unbacked claim in that field is a corruption "
                f"signal: fix or remove it deliberately. Nothing was written.")
        return None, None
    # PROTECTIVE POLARITY (see the polarity note above ``_db_kind_protective``): a recorded
    # provenance string may name a path from another machine, and an unclassifiable one must
    # read as "real" so the directory is protected HARDER, never as a permit.
    derived = _db_kind_protective(prior_db)
    if stored is not None and stored != derived:
        raise MaterializeRefusal(
            f"{where}: REFUSING — its recorded provenance DISAGREES WITH ITSELF: it "
            f"stores db_kind={stored!r} while its universe_db={prior_db!r} re-derives as "
            f"{derived!r}. That field is exactly what the fixture-over-real refusal is "
            f"computed from, so a manifest that lies in it could authorize the one "
            f"overwrite no flag may authorize. An internally inconsistent manifest is "
            f"hand-edited or corrupt: fix or remove it deliberately. Nothing was written.")
    return prior_db, derived


def _read_manifest_provenance(out_root, who="materialize"):
    """Read a pre-existing ``out_root``'s recorded provenance, for DISCLOSURE. Returns
    ``(provenance_dict_or_None, human_note)``. An absent manifest is NOT an error — it
    just means no provenance was recorded. File-type/parse guarding is the shared
    ``_load_json_regular_file`` reader's job. ``who`` names the calling writer in the
    refusal (``materialize``)."""
    manifest_path = os.path.join(out_root, "fold_manifest.json")
    try:
        os.lstat(manifest_path)
    except OSError:
        return None, "no fold_manifest.json present — no provenance recorded"
    existing = _load_json_regular_file(
        manifest_path, MaterializeRefusal,
        f"{who}: refusing to write into {_rel(out_root)} — its fold_manifest.json")
    if not isinstance(existing, dict):
        # Belt only: the shared reader now enforces ``require=dict`` and refuses a
        # list/null/string manifest itself (MAJOR fix), so this branch is unreachable
        # through it. Kept so the function stays correct if ever handed a laxer reader.
        return None, "fold_manifest.json is not a JSON object — no provenance recorded"
    prov = existing.get(_PROVENANCE_KEY)
    if isinstance(prov, dict):
        return prov, f"recorded {prov.get('generated', '?')} from {prov.get('universe_db')!r}"
    # GRANDFATHERING (the B2 lockout fix): manifests written before TASK_904 carry no
    # _provenance block, but they DO carry universe_db/fold_set — enough to run the
    # fixture-over-real check on the tracked folds/real/ tree that predates this guard.
    legacy = {k: existing.get(k) for k in ("universe_db", "fold_set", "generated")
              if existing.get(k) is not None}
    if legacy:
        legacy["legacy"] = True
        return legacy, (f"pre-TASK_904 manifest (no {_PROVENANCE_KEY} block) — "
                        f"reconstructed from universe_db={legacy.get('universe_db')!r}, "
                        f"generated {legacy.get('generated', '?')}")
    return None, "fold_manifest.json carries no provenance fields"


def _assert_out_root_writable(out_root, db_path, fold_set, allow_overwrite):
    """TASK_904A gate — runs BEFORE any write in ``materialize``.

    Checked in this order — DESTINATION FIRST, because the destination's identity is the
    only fact here that no caller can misreport:

    * unusable ``db_path`` (falsy / not a regular file) -> refused unconditionally, before
      anything is classified, so no refusal below can be made inert by it;
    * the CANONICAL tracked real-fold tree -> refused unless the FULL, mutually consistent
      unlock is present (``--fold-set real`` AND a real-classed db AND the flag). This runs
      even when the tree is EMPTY, and consults nothing inside it. See
      ``_authorize_canonical_write``;
    * absent / empty ``out_root``  -> proceed, no flag needed (first materialization);
    * non-empty ``out_root``       -> requires the caller's explicit ``allow_overwrite``
      (CLI ``--overwrite-existing``). The refusal names the directory and the flag, and
      leaves the directory byte-for-byte untouched;
    * fixture-over-real            -> refused regardless of the flag (see the design note
      above). The PRIOR side is the PROTECTIVE polarity, RE-DERIVED from the recorded
      ``universe_db`` (never read out of ``db_kind``); THIS side is the AUTHORIZING
      polarity, so an unprovable ``--db`` refuses here exactly like the fixture does;
    * self-contradictory provenance -> refused regardless of the flag.
    With the opt-in given, the recorded provenance is PRINTED and compared against this
    call; disagreements are announced loudly and proceed. Returns the disclosed
    provenance dict (or None)."""
    _assert_db_usable(db_path, "materialize")
    _authorize_canonical_write(out_root, db_path, fold_set, allow_overwrite, "materialize")
    if not os.path.isdir(out_root) or not os.listdir(out_root):
        return None                      # nothing there to protect
    prov, note = _read_manifest_provenance(out_root)   # may itself refuse (type/parse)
    # F1: the class is re-derived from universe_db, NEVER read out of db_kind — the
    # refusal below is computed from it, so trusting the stored string would let a
    # directory authorize its own clobbering.
    prior_db, prior_kind = _prior_db_kind(prov, f"materialize: {_rel(out_root)}")
    # THIS call's side of the comparison is an AUTHORIZING question ("may this db overwrite
    # a real-provenance tree?"), so it takes the authorizing polarity: UNPROVEN refuses like
    # FIXTURE. ``this_cls`` is the tri-state value, carried only so the DISCLOSURE line below
    # reports what was actually determined — the cycle-2 diagnostic printed "(real -> real)"
    # about the project fixture itself.
    this_cls, _this_why = _db_class(db_path, out_root)
    this_not_real = _require_positively_real_db(db_path, out_root, "materialize")

    # The DESTINATION-IDENTITY fence already ran at the top of this function (and again,
    # as the write scope, in ``materialize``) — it is not repeated here, because it must
    # not depend on anything read out of the directory. What remains below is the class
    # comparison, which still correctly catches every OTHER real-provenance fold tree.
    if prior_kind == "real" and this_not_real is not None:
        raise MaterializeRefusal(
            f"materialize: REFUSING to overwrite {_rel(out_root)} — its recorded "
            f"provenance says it was materialized from a REAL database ({prior_db!r}) "
            f"and this call's --db ({_rel(db_path)}) is NOT POSITIVELY PROVEN real-classed "
            f"({this_cls}): {this_not_real}. Downgrading real fold rails to fixture-derived "
            f"— or to unprovable — content is never an intended overwrite, so "
            f"{_OVERWRITE_FLAG} does NOT authorize it. "
            f"Point --out-root at a scratch directory instead.")
    if not allow_overwrite:
        raise MaterializeRefusal(
            f"materialize: refusing to write into non-empty {_rel(out_root)} — it "
            f"already holds {len(os.listdir(out_root))} entr"
            f"{'y' if len(os.listdir(out_root)) == 1 else 'ies'} (provenance: {note}). "
            f"Re-materializing over an existing fold tree is an explicit act: pass "
            f"{_OVERWRITE_FLAG} (CLI) / allow_overwrite=True (API) to authorize it, or "
            f"point --out-root at a fresh directory. Nothing was written.")
    print(f"[837][overwrite] {_OVERWRITE_FLAG} given: re-materializing over existing "
          f"content in {_rel(out_root)} (provenance: {note})")
    disagreements = []
    if prior_db and prior_db != _rel(db_path):
        disagreements.append(f"db: recorded {prior_db!r} -> this call "
                             f"{_rel(db_path)!r} ({prior_kind} -> {this_cls})")
    prior_set = (prov or {}).get("fold_set")
    if prior_set is not None and prior_set != fold_set:
        disagreements.append(f"fold_set: recorded {prior_set!r} -> this call {fold_set!r}")
    prior_hash = (prov or {}).get("content_sha256")
    if prior_hash:
        now_hash = _hash_tree(out_root, exclude=("fold_manifest.json",))
        if now_hash != prior_hash:
            disagreements.append(
                f"content: the directory has been MODIFIED since it was materialized "
                f"(recorded {prior_hash[:12]}... != current {now_hash[:12]}...)")
    for d in disagreements:
        print(f"[837][overwrite] PROVENANCE DISAGREEMENT — {d}")
    if not disagreements:
        print("[837][overwrite] provenance AGREES with this call (ordinary "
              "re-materialization)")
    return prov


def _hash_tree(root, exclude=()):
    """Combined content hash of every file under ``root`` (path-and-bytes). Used both to
    prove a refused write left a directory byte-for-byte untouched, and to record what a
    materialization produced. ``exclude`` names root-relative paths to skip (the manifest
    excludes ITSELF, since it is what carries the hash)."""
    h = hashlib.sha256()
    skip = set(exclude)
    for dirpath, _dirnames, filenames in sorted(os.walk(root)):
        for name in sorted(filenames):
            p = os.path.join(dirpath, name)
            rel = os.path.relpath(p, root)
            if rel in skip:
                continue
            h.update(rel.encode())
            with open(p, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()


# =============================================================================
# date parsing — BOTH conventions, used conservatively
# =============================================================================

def utc_epoch(d):
    """UTC-midnight parse (trader.backtest._to_epoch — daily-store timestamps are UTC)."""
    return backtest._to_epoch(d)


def loader_epoch(d):
    """The training loader's parse (VectorBTTradingEnv._load_database_prices; the 835
    mirror). TASK_869: the loader's parse is now UTC-midnight (was local-timezone), so this
    is IDENTICAL to ``utc_epoch`` — the conservative min/max helpers below collapse to one
    bound, which is the point of the re-base. Loader bounds are INCLUSIVE:
    ``timestamp >= start AND timestamp <= end`` (vectorbt_env.py:660)."""
    return int(datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def _max_end_epoch(d):
    """Conservative LATEST instant a date string can mean under either parse."""
    return max(utc_epoch(d), loader_epoch(d))


def _min_start_epoch(d):
    """Conservative EARLIEST instant a date string can mean under either parse."""
    return min(utc_epoch(d), loader_epoch(d))


def _date_add(d, days):
    return (date.fromisoformat(d) + timedelta(days=days)).isoformat()


def _rel(p):
    """Repo-relative display path (committed artifacts never embed absolute paths)."""
    p = os.path.abspath(p)
    r = os.path.relpath(p, _REPO)
    return p if r.startswith("..") else r


def _ro_connect(db_path):
    return sqlite3.connect(f"file:{os.path.abspath(db_path)}?mode=ro", uri=True)


def _write_json(path, obj):
    _guard_write_path(path)      # TASK_904A: THE shared destination-identity choke point
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=float)
    return path


def _f(x):
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


# =============================================================================
# PIT universe — time-bounded re-implementation of build_pit_universe's rule
# =============================================================================

def derive_pit_universe(db_path, as_of, n_assets=PIT_N_ASSETS, min_days=PIT_MIN_DAYS):
    """Universe as-of ``as_of`` (train-end) from data <= as_of ONLY.

    Rule (pre-registered; SPEC §PIT): symbol contains '/', quote == USD, base not in
    STABLE_FIAT (== trader/data.py:401 build_pit_universe's filter), coverage floor
    ``min_days`` DAILY bars with timestamp <= as_of (UTC midnight), ranked by that count
    DESC (tie: asset_id ASC), top ``n_assets``. LOUD DEVIATION from build_pit_universe:
    daily-bar counts (``ohlcv_daily``) instead of its unbounded 1-min counts — see module
    docstring #2 and SPEC.md. Read-only; never writes."""
    as_of_ts = utc_epoch(as_of)
    con = _ro_connect(db_path)
    rows = con.execute(
        """SELECT a.id, a.symbol, COUNT(*) AS nd
           FROM ohlcv_daily o JOIN assets a ON o.asset_id = a.id
           WHERE o.timestamp <= ?
           GROUP BY a.id""", (as_of_ts,)).fetchall()
    con.close()
    qual = []
    for aid, sym, nd in rows:
        if not sym or "/" not in sym:
            continue
        base, quote = sym.split("/", 1)
        if quote.upper() != "USD" or base.upper() in STABLE_FIAT:
            continue
        if nd >= min_days:
            qual.append((int(nd), int(aid), sym))
    qual.sort(key=lambda t: (-t[0], t[1]))
    sel = qual[:n_assets]
    if not sel:
        raise LeakError(
            f"PIT universe EMPTY as-of {as_of} on {_rel(db_path)} (rule min_days="
            f"{min_days}) — refusing to emit a fold without a point-in-time universe. "
            f"(On a minute-only db slice this is expected; materialize real folds "
            f"against a db whose ohlcv_daily covers the train window.)")
    return {
        "as_of": as_of,
        "as_of_ts_utc": as_of_ts,
        "rule": {
            "source_table": "ohlcv_daily",
            "filter": "symbol has '/', quote==USD, base not in trader.config.STABLE_FIAT "
                      "(build_pit_universe's filter, time-bounded at as_of)",
            "min_days": min_days,
            "rank": "daily-bar count <= as_of DESC, tie asset_id ASC",
            "n_assets_cap": n_assets,
            "deviation": "trader/data.py:401 build_pit_universe has NO as-of parameter "
                         "(counts ALL 1-min bars) — rule re-implemented here time-bounded "
                         "on the canonical daily store; trader/data.py untouched.",
        },
        "n_qualifying": len(qual),
        "n_selected": len(sel),
        "symbols": [s for (_, _, s) in sel],
        "asset_ids": [a for (_, a, _) in sel],
        "counts": {s: n for (n, _, s) in sel},
        "db": _rel(db_path),
    }


def load_pinned_universe():
    """The pinned 64-asset 821 universe, byte-order preserved (TASK_740 landmine: never
    reorder). Emitted only under the LEAKY-UNIVERSE label — selected with knowledge
    through 2025, survivorship-leaky for fold evals."""
    with open(PINNED_PERIODS) as fh:
        return json.load(fh)["periods"][0]["assets"]


# =============================================================================
# fold materialization — periods-files (821 format) + universes + job manifests
# =============================================================================

def _periods_file(name, start, end, assets, comment):
    """EXACT 821 schema: consumers (train_jax_fast.py:223-231, 835 eval_vector.py
    --periods-file) read periods[0] {start_date, end_date, assets} only."""
    return {"_comment": comment,
            "periods": [{"name": name, "start_date": start, "end_date": end,
                         "assets": list(assets)}]}


def _check_fold_definition(fold):
    """Fail loud BEFORE writing anything if a fold definition itself is malformed
    (the harness implements pre-registered splits; it never repairs them)."""
    ts0, ts1 = fold["train"]
    es0, es1 = fold["eval"]
    if not (ts0 < ts1 and es0 < es1):
        raise LeakError(f"{fold['fold_id']}: degenerate window (train {ts0}..{ts1}, "
                        f"eval {es0}..{es1})")
    if not _max_end_epoch(ts1) < _min_start_epoch(es0):
        raise LeakError(f"{fold['fold_id']}: train end {ts1} does not strictly precede "
                        f"eval start {es0} under the conservative dual parse")


def materialize(db_path, folds, fold_set, out_root, n_assets, min_days,
                jobs_db, seeds=SEEDS, verify_minute=True, allow_overwrite=False):
    """The gated entry point (see ``_materialize_writes`` for the writing itself).

    TASK_904A: the whole write phase runs inside ``_canonical_write_scope``, so if
    ``out_root`` IS the canonical tracked real-fold tree the destination-identity unlock
    must be affirmatively complete BEFORE anything is written, and the shared choke point
    (``_guard_write_path``) is opened for that subtree only, for this call only."""
    with _canonical_write_scope(out_root, db_path, fold_set, allow_overwrite,
                                who="materialize"):
        _assert_out_root_writable(out_root, db_path, fold_set, allow_overwrite)
        return _materialize_writes(db_path, folds, fold_set, out_root, n_assets,
                                   min_days, jobs_db, seeds, verify_minute)


def _materialize_writes(db_path, folds, fold_set, out_root, n_assets, min_days,
                        jobs_db, seeds=SEEDS, verify_minute=True):
    """Emit all fold artifacts under ``out_root``:
      folds/<id>/universe_pit.json, period_train.json, period_eval.json,
      period_train_leaky.json, period_eval_leaky.json
      out_root/train_jobs.manifest, eval_jobs.manifest, fold_manifest.json
    Returns the fold_manifest dict. Read-only db access; validation is a SEPARATE pass
    (validate_folds) that re-parses these files — never trusts this construction.

    TASK_904A: writing into a NON-EMPTY ``out_root`` requires ``allow_overwrite=True``
    (CLI ``--overwrite-existing``) — the operator's intent is the gate. The directory's
    recorded provenance is disclosed, not trusted as a credential; the provenance-driven
    REFUSALS (none flag-overridable) are the canonical-tree identity fence,
    fixture-over-real, and a provenance block that contradicts itself. See
    ``_assert_out_root_writable``. Called ONLY through ``materialize`` above, which holds
    the gate and the write scope."""
    pinned = load_pinned_universe()
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    fold_records, train_lines, eval_lines, notes = [], [], [], []

    for fold in folds:
        _check_fold_definition(fold)
        fid = fold["fold_id"]
        (t0, t1), (e0, e1) = fold["train"], fold["eval"]
        fdir = os.path.join(out_root, fid)
        os.makedirs(fdir, exist_ok=True)

        uni = derive_pit_universe(db_path, as_of=t1, n_assets=n_assets, min_days=min_days)
        uni.update({"fold_id": fid, "generated": generated})
        if verify_minute:
            uni["minute_coverage_check"] = _minute_coverage_note(
                db_path, uni["asset_ids"], uni["as_of_ts_utc"])
        upath = _write_json(os.path.join(fdir, "universe_pit.json"), uni)

        c_pit = (f"TASK_837 {fid} — POINT-IN-TIME universe as-of train-end {t1} "
                 f"(rule: universe_pit.json; NO data past {t1} used). "
                 f"821 periods[0] format; consumers read periods[0] only.")
        c_leaky = (f"TASK_837 {fid} — *** LEAKY-UNIVERSE *** pinned 64-asset 821 list "
                   f"(selected with knowledge through 2025 — survivorship leak for fold "
                   f"evals; emitted ONLY for continuity with 821-lineage numbers, order "
                   f"byte-identical, TASK_740 landmine: never reorder).")
        files = {
            "universe": upath,
            "train_pit": _write_json(os.path.join(fdir, "period_train.json"),
                                     _periods_file(f"837_{fid}_train_{t0}_{t1}", t0, t1,
                                                   uni["symbols"], c_pit)),
            "eval_pit": _write_json(os.path.join(fdir, "period_eval.json"),
                                    _periods_file(f"837_{fid}_eval_{e0}_{e1}", e0, e1,
                                                  uni["symbols"], c_pit)),
            "train_leaky": _write_json(os.path.join(fdir, "period_train_leaky.json"),
                                       _periods_file(f"837_{fid}_train_LEAKY_{t0}_{t1}",
                                                     t0, t1, pinned, c_leaky)),
            "eval_leaky": _write_json(os.path.join(fdir, "period_eval_leaky.json"),
                                      _periods_file(f"837_{fid}_eval_LEAKY_{e0}_{e1}",
                                                    e0, e1, pinned, c_leaky)),
        }

        train_jobs, eval_jobs = [], []
        for s in seeds:
            tj = f"837_{fid}_{ARM['name']}_s{s}"
            cmd = "python -u train_jax_fast.py " + ARM["train_flags"].format(
                db=jobs_db, pf=_rel(files["train_pit"]), seed=s, name=tj)
            train_lines.append(f"{tj} gpu 32G {cmd}")
            train_jobs.append(tj)
            ej = f"837_{fid}_eval_s{s}"
            ecmd = ("python -u experiments/835_stage4_eval/eval_vector.py "
                    + ARM["eval_flags"].format(
                        db=jobs_db, pf=_rel(files["eval_pit"]),
                        n_assets=uni["n_selected"],
                        ckpt=f"jax_ckpt_fast/{tj}", tag=ej,
                        out_dir=f"runs/837_walkforward/{fold_set}/{fid}"))
            eval_lines.append(f"{ej} gpu 32G {ecmd}")
            eval_jobs.append(ej)

        fold_records.append({
            "fold_id": fid, "train_window": [t0, t1], "eval_window": [e0, e1],
            "files": {k: _rel(v) for k, v in files.items()},
            "universe": {"as_of": t1, "n_selected": uni["n_selected"],
                         "n_qualifying": uni["n_qualifying"]},
            "jobs": {"train": train_jobs, "eval": eval_jobs},
        })

    hdr = [f"# TASK_837 walk-forward {fold_set} folds — generated {generated}",
           f"# arm mirrors experiments/828_stage3_train/jobs_phase1.manifest "
           f"(cost0 phase-1; fee16 continuation = planner phase-2+ decision)",
           f"# consumable by tools/deploy/run_queue.sh: <job_id> <gpu|cpu> <mem> <command>"]
    # These two are the only writes in the module that do not go through _write_json, so
    # they call the shared choke point explicitly (defence in depth — materialize's write
    # scope has already authorized this destination if it is the canonical tree).
    tman = _guard_write_path(os.path.join(out_root, "train_jobs.manifest"))
    with open(tman, "w") as fh:
        fh.write("\n".join(hdr + train_lines) + "\n")
    eman = _guard_write_path(os.path.join(out_root, "eval_jobs.manifest"))
    with open(eman, "w") as fh:
        fh.write("\n".join(hdr + ["# eval operator carried from the training arm "
                                  "(hold_bias 0.0 explicit — 835's CLI default 3.0 is "
                                  "the jobs.manifest arm, wrong here)"] + eval_lines) + "\n")

    manifest = {
        "task": "TASK_837", "fold_set": fold_set, "generated": generated,
        "universe_db": _rel(db_path), "jobs_db": jobs_db,
        "n_assets": n_assets, "min_days": min_days, "seeds": list(seeds),
        "folds": fold_records,
        "manifests": {"train": _rel(tman), "eval": _rel(eman)},
        "notes": notes + [
            "validate with: python experiments/837_walkforward/walkforward.py validate "
            f"--fold-root {_rel(out_root)} --db <db>",
            "REAL-fold universes derived from the fixture's full ohlcv_daily panel "
            "(fixture daily == db daily, LEDGER 830/E0-REAL); the scheduled "
            "re-materialization against trading_data.db must reproduce them "
            "content-identically (M4 identity contract, SPEC §PIT: periods-files "
            "byte-identical; universe_pit.json identical on the derivation fields, "
            "volatile generated/db/minute_coverage_check excluded)."
            if fold_set == "real" else
            "smoke fold set — never run these jobs; rails proof only.",
        ],
    }
    # TASK_904A provenance — DISCLOSURE, not a credential. Recorded LAST so the content
    # hash covers everything this call wrote EXCEPT the manifest that carries it.
    manifest[_PROVENANCE_KEY] = {
        "materializer": _MATERIALIZER_ID,
        "universe_db": _rel(db_path),
        # PROTECTIVE polarity: this string is a DISCLOSURE record for humans and for the
        # next call's "should I protect this directory harder?" question, never a permit.
        "db_kind": _db_kind_protective(db_path),
        "fold_set": fold_set,
        "generated": generated,
        "content_sha256": _hash_tree(out_root, exclude=("fold_manifest.json",)),
        "note": "DISCLOSURE ONLY (TASK_904A): records what produced this directory so a "
                "later re-materialization can say out loud what it is replacing. It "
                "grants no write permission — that comes ONLY from the caller's "
                "explicit --overwrite-existing / allow_overwrite=True — and it is not a "
                "credential to be matched. The refusals it can trigger (none "
                "flag-overridable) are computed from universe_db/the canonical-tree "
                "identity, never from this db_kind string (db_kind is recorded for "
                "humans only).",
    }
    _write_json(os.path.join(out_root, "fold_manifest.json"), manifest)
    print(f"[837] materialized {len(fold_records)} {fold_set} fold(s) -> {_rel(out_root)}")
    return manifest


def _minute_coverage_note(db_path, asset_ids, as_of_ts):
    """Cross-check (SPEC §PIT): PIT members should have SOME minute coverage <= as_of,
    because the training loader reads the 1-min ``ohlcv`` table. On a db whose minute
    slice starts after as_of (the fixture, for the real folds) the check is VACUOUS and
    says so loudly instead of passing silently."""
    con = _ro_connect(db_path)
    any_min = con.execute("SELECT COUNT(*) FROM ohlcv WHERE timestamp <= ?",
                          (as_of_ts,)).fetchone()[0]
    if any_min == 0:
        con.close()
        return {"status": "VACUOUS", "detail":
                "db has ZERO minute bars <= as_of — minute-coverage check impossible "
                "here; re-run on the scheduled materialization against trading_data.db."}
    ph = ",".join("?" * len(asset_ids))
    cov = dict(con.execute(
        f"SELECT asset_id, COUNT(*) FROM ohlcv WHERE asset_id IN ({ph}) "
        f"AND timestamp <= ? GROUP BY asset_id", (*asset_ids, as_of_ts)).fetchall())
    con.close()
    missing = [a for a in asset_ids if cov.get(a, 0) == 0]
    return {"status": "CHECKED", "n_members_without_minute_bars": len(missing),
            "asset_ids_without_minute_bars": missing}


# =============================================================================
# leak-free gates AS CODE — parse the generated files, never trust construction
# =============================================================================

def _read_periods(path):
    with open(path) as fh:
        p = json.load(fh)["periods"][0]
    return p["start_date"], p["end_date"], p["assets"]


def _pf_from_cmd(cmd):
    toks = cmd.split()
    return toks[toks.index("--periods-file") + 1] if "--periods-file" in toks else None


# =============================================================================
# TASK_905 — resolve fold artifacts against the tree ``validate`` is ACTUALLY pointed
# at, not against ``_REPO``
# =============================================================================
# The defect (measured, factory run 41): every path recorded in fold_manifest.json /
# the job manifests is a REPO-RELATIVE string written at materialize time (``_rel(v)``);
# ``validate`` used to re-join every one of them onto this script's OWN ``_REPO``, so
# ``validate --fold-root <copy>`` silently re-read the ORIGINAL tree no matter what
# directory it was pointed at — a `cp -r`'d copy, a restored backup, a staged copy on
# gpu-server all validated the WRONG FILES with a clean exit 0.
#
# Decision (scope item 1, recorded here per the task banner requirement): resolve
# relative to ``--fold-root`` itself (option a), not option b ("keep _REPO resolution,
# refuse when --fold-root disagrees with the manifest's own tree"). Option b closes the
# false PASS too, but leaves the tool unusable for the exact audit case ``--fold-root``
# exists to serve — a relocated tree is, by definition, not the tree the manifest
# describes, so b's refusal would fire on every legitimate use. Materialize always
# writes per-fold artifacts at ``<out_root>/<fold_id>/<basename>`` and the two job
# manifests at ``<out_root>/<basename>`` (``_materialize_writes``), so the RECORDED
# path's prefix is discarded entirely and rebuilt from ``fold_root`` + ``fold_id`` (or
# ``fold_root`` alone) + the recorded basename — this resolves correctly under ANY
# relocation, in-repo or not, without needing to know the tree's original location.
#
# Scope item 2 (fail-closed, not optional): a resolved path must never land outside the
# named ``fold_root`` — not via a literal ``..`` in a recorded string, and not via a
# tampered ``fold_id`` (attacker-controlled JSON) or a symlink planted inside the copy.
# ``_require_within_fold_root`` checks containment on ``os.path.realpath`` of BOTH
# sides (this repo has been bitten twice by realpath-vs-abspath divergence —
# TASK_886/888 — a bare ``startswith`` on unnormalized strings misses a symlink escape
# even when it catches a ``..`` escape) and raises loud, never redirects silently.
#
# It raises so that the REFUSAL happens strictly BEFORE the caller can ``open()`` the
# out-of-root path — that ordering is the whole guarantee and is unchanged. But its
# callers inside ``validate_folds`` CATCH it and funnel it through ``rec()`` onto the
# aggregate ``fails`` list, exactly like every other check, so the scan still completes
# and the single terminal raise carries EVERY violation (panel FINDING 1). The escape is
# not softened by this: a recorded failure is still a non-empty ``fails`` list, still a
# terminal ``LeakError``, still exit 1 — and the out-of-root bytes are still never read.
def _require_within_fold_root(fold_root, path, what):
    root_r = os.path.realpath(fold_root)
    p_r = os.path.realpath(path)
    if not (p_r == root_r or p_r.startswith(root_r + os.sep)):
        raise LeakError(
            f"validate: {what} resolves to {p_r}, OUTSIDE --fold-root {root_r} — "
            f"refusing rather than silently reading a different tree (TASK_905 fail-"
            f"closed escape guard)")
    return p_r


def _resolve_fold_artifact(fold_root, fid, recorded, what):
    """Resolve a per-fold artifact path RECORDED in fold_manifest.json (e.g.
    ``f["files"]["train_pit"]`` or a job manifest's ``--periods-file``) against
    ``fold_root`` — the tree actually being validated — via ``fold_root/fid/basename``.
    See the module note above for why (TASK_905)."""
    p = os.path.join(fold_root, fid, os.path.basename(recorded))
    return _require_within_fold_root(fold_root, p, what)


def _resolve_root_artifact(fold_root, recorded, what):
    """Same as ``_resolve_fold_artifact`` for artifacts recorded directly under
    ``fold_root`` with no ``fold_id`` subdirectory (the two job manifests)."""
    p = os.path.join(fold_root, os.path.basename(recorded))
    return _require_within_fold_root(fold_root, p, what)


def validate_folds(fold_root, db_path, verbose=True):
    """THE leak gates (module docstring ASSERT-1 / ASSERT-2 / POLICY-PIN). Parses
    fold_manifest.json + every generated periods-file/universe/manifest/corridor under
    ``fold_root`` and re-checks them against the db AND against the PRE-REGISTERED
    POLICY constants (never against the artifacts' own tamperable rule blocks). Raises
    LeakError listing EVERY violation (fail loud); returns the check records on
    success."""
    man_path = os.path.join(fold_root, "fold_manifest.json")
    if not os.path.lexists(man_path):
        raise LeakError(f"no fold_manifest.json under {_rel(fold_root)}")
    # F3: the SHARED guarded reader. `validate` runs at the tail of every `materialize`
    # CLI call and standalone; a bare open() here hung forever on a FIFO manifest.
    man = _load_json_regular_file(man_path, LeakError,
                                  f"validate: {_rel(man_path)}")
    con = _ro_connect(db_path)
    checks, fails = [], []

    # FINDING 3 (panel, correctness NIT): the read-only sqlite handle used to
    # leak on EVERY early raise out of this body — the unguarded corridor-JSON
    # parse below could already reach `con.close()`-never-runs, and Finding 1's
    # aggregate-raise fix must not add a second trigger. One try/finally closes
    # both, and keeps the aggregate raise itself OUTSIDE the guarded region so
    # the LeakError still propagates unchanged.
    try:

        def rec(fold, name, ok, detail):
            checks.append({"fold": fold, "check": name, "ok": bool(ok), "detail": detail})
            if not ok:
                fails.append(f"[{fold}] {name}: {detail}")
            if verbose:
                print(f"  [{'PASS' if ok else 'FAIL'}] {fold} :: {name} — {detail}")

        pinned = load_pinned_universe()
        fold_train_files = {}
        fold_uni_syms = {}

        # ---- POLICY-PIN (MAJOR-1): everything below is checked against constants ----
        fold_set = man.get("fold_set")
        policy = POLICY.get(fold_set)
        if policy is None:
            rec("*", "POLICY-PIN fold set", False,
                f"unknown fold_set '{fold_set}' — only the pre-registered sets "
                f"{sorted(POLICY)} validate (the harness implements splits, never invents)")
        else:
            man_ids = sorted(f["fold_id"] for f in man["folds"])
            rec("*", "POLICY-PIN fold-set completeness", man_ids == sorted(policy["folds"]),
                f"manifest folds {man_ids} == pre-registered {sorted(policy['folds'])}")
            rec("*", "POLICY-PIN manifest rule params",
                man.get("n_assets") == policy["n_assets"]
                and man.get("min_days") == policy["min_days"],
                f"manifest n_assets={man.get('n_assets')} min_days={man.get('min_days')} == "
                f"pre-registered {policy['n_assets']}/{policy['min_days']}")

        for f in man["folds"]:
            fid = f["fold_id"]
            # FINDING 1 (panel, correctness lens): the containment guard REFUSES before
            # any open() — that property is load-bearing and unchanged — but the refusal
            # is RECORDED like every other check instead of raising out of this loop.
            # Raising here made the audit order-dependent and silently incomplete:
            # violations already collected for earlier folds were discarded from the
            # raised message (the only thing the CLI puts on stderr, hence the only thing
            # a CI grep sees), and folds AFTER the escaping one were never evaluated at
            # all. `validate_folds` promises "EVERY violation"; an escape in a partially
            # restored backup is exactly when the operator most needs the other findings.
            try:
                files = {k: _resolve_fold_artifact(fold_root, fid, v, f"{fid} files.{k}")
                         for k, v in f["files"].items()}
            except LeakError as e:
                rec(fid, "artifacts resolve INSIDE --fold-root", False, str(e))
                continue
            try:
                tr0, tr1, tr_assets = _read_periods(files["train_pit"])
                ev0, ev1, ev_assets = _read_periods(files["eval_pit"])
                lt0, lt1, lt_assets = _read_periods(files["train_leaky"])
                le0, le1, le_assets = _read_periods(files["eval_leaky"])
            except Exception as e:  # unparseable artifact = leak-gate failure, not a crash
                rec(fid, "artifacts-parse", False, f"unreadable periods file: {e}")
                continue
            # TASK_905: compare the RECORDED strings directly (both files.* and a job's
            # --periods-file were written by the same `_rel(v)` call at materialize time),
            # never a path recomputed relative to _REPO — that recomputation is exactly
            # what silently diverged for any tree materialized outside the repo (the two
            # spellings agree only in-repo; see the CLI's own note on this at the smoke
            # real-fold scratch dir). Membership is about recorded IDENTITY, not about
            # where the bytes physically sit.
            fold_train_files[fid] = {f["files"]["train_pit"], f["files"]["train_leaky"]}

            # ---- POLICY-PIN (MAJOR-1): parsed windows == the pre-registered constants ----
            if policy is not None:
                pol = policy["folds"].get(fid)
                rec(fid, "POLICY-PIN fold windows pre-registered",
                    pol is not None and (tr0, tr1) == tuple(pol["train"])
                    and (ev0, ev1) == tuple(pol["eval"]),
                    (f"parsed train {tr0}..{tr1} / eval {ev0}..{ev1} == pre-registered "
                     f"{pol['train'][0]}..{pol['train'][1]} / {pol['eval'][0]}..{pol['eval'][1]}"
                     if pol else f"fold_id '{fid}' is NOT in the pre-registered {fold_set} set"))

            # ---- ASSERT-1: no train/eval overlap, from the PARSED files, dual parse ----
            t_end = max(_max_end_epoch(tr1), _max_end_epoch(lt1))
            e_start = min(_min_start_epoch(ev0), _min_start_epoch(le0))
            rec(fid, "ASSERT-1 train/eval windows disjoint", t_end < e_start,
                f"parsed train end {tr1} (<= {t_end}) vs eval start {ev0} (>= {e_start}); "
                f"loader bounds are end-INCLUSIVE (vectorbt_env.py:660)")
            rec(fid, "windows well-formed",
                tr0 < tr1 and ev0 < ev1 and (tr0, tr1) == (lt0, lt1) and (ev0, ev1) == (le0, le1),
                f"train {tr0}..{tr1}, eval {ev0}..{ev1}, leaky dates match pit dates")

            # ---- ASSERT-2a: what a training loader WOULD ingest from the train file ----
            s_ts = min(_min_start_epoch(tr0), _min_start_epoch(lt0))
            e_ts = t_end
            for table in ("ohlcv", "ohlcv_daily"):
                mx = con.execute(f"SELECT MAX(timestamp) FROM {table} "
                                 f"WHERE timestamp >= ? AND timestamp <= ?",
                                 (s_ts, e_ts)).fetchone()[0]
                if mx is None:
                    detail = (f"VACUOUS — db has no {table} rows in the parsed train bounds "
                              f"(loader would load nothing here)")
                    if table == "ohlcv" and fold_set == "real":
                        detail += ("; *** WARNING: these manifests are LAUNCH-READY against "
                                   "trading_data.db, where this assert HAS teeth — run "
                                   "`validate --db trading_data.db` (read-only) BEFORE "
                                   "launching train jobs (SPEC pre-flight) ***")
                        print(f"  [WARN] {fid} :: ASSERT-2 ingest-max ohlcv is VACUOUS on "
                              f"this db — pre-flight `validate --db trading_data.db` "
                              f"required before any fold training launch")
                    rec(fid, f"ASSERT-2 ingest-max {table}", True, detail)
                else:
                    rec(fid, f"ASSERT-2 ingest-max {table}", mx < e_start,
                        f"max ingestable ts {datetime.fromtimestamp(mx, timezone.utc):%Y-%m-%d %H:%M} "
                        f"UTC < eval start (>= {e_start})")

            # ---- ASSERT-2b: PIT universe re-derivation from data <= train-end ONLY,
            # with the rule parameters PINNED to POLICY (MAJOR-1: the artifact's own rule
            # block is tamperable and is itself checked, never trusted) ----
            try:
                with open(files["universe"]) as fh:
                    uni = json.load(fh)
                exp_n = policy["n_assets"] if policy else uni["rule"]["n_assets_cap"]
                exp_min = policy["min_days"] if policy else uni["rule"]["min_days"]
                if policy is not None:
                    rec(fid, "POLICY-PIN universe rule",
                        uni["rule"]["n_assets_cap"] == exp_n
                        and uni["rule"]["min_days"] == exp_min,
                        f"artifact rule cap={uni['rule']['n_assets_cap']} min_days="
                        f"{uni['rule']['min_days']} == pre-registered {exp_n}/{exp_min} "
                        f"(re-derivation params are PINNED, never read from the artifact)")
                ok_asof = uni["as_of"] == tr1
                re_uni = derive_pit_universe(db_path, as_of=tr1, n_assets=exp_n,
                                             min_days=exp_min)
                same = re_uni["symbols"] == uni["symbols"]
                rec(fid, "ASSERT-2 PIT universe re-derivation", ok_asof and same,
                    f"as_of=={tr1}: {ok_asof}; re-derived {re_uni['n_selected']} symbols "
                    f"{'IDENTICAL (incl. order)' if same else 'MISMATCH — tampered or derived with post-train-end data'}")
                rec(fid, "periods-files carry the PIT universe",
                    tr_assets == uni["symbols"] and ev_assets == uni["symbols"],
                    "train/eval pit files carry universe_pit.json symbols verbatim "
                    "(same order train==eval — TASK_740 landmine)")
                fold_uni_syms[fid] = uni["symbols"]
            except LeakError as e:
                rec(fid, "ASSERT-2 PIT universe re-derivation", False, str(e))
            rec(fid, "leaky files carry the pinned 64 byte-identically",
                lt_assets == pinned and le_assets == pinned,
                "LEAKY-UNIVERSE label + pinned order preserved")
            rec(fid, "leaky files labeled LOUDLY",
                "LEAKY-UNIVERSE" in json.load(open(files["train_leaky"]))["_comment"]
                and "LEAKY-UNIVERSE" in json.load(open(files["eval_leaky"]))["_comment"],
                "_comment contains LEAKY-UNIVERSE")

            # ---- corridor artifact bounds (if computed): never loaded past eval-end ----
            cpath = os.path.join(os.path.dirname(files["universe"]), "corridor.json")
            if os.path.exists(cpath):
                with open(cpath) as fh:
                    cor = json.load(fh)
                ok = (cor["panel_window"][1] == ev1
                      and cor["panel_last_day_ts"] < _min_start_epoch(_date_add(ev1, 1))
                      and cor["eval_window"] == [ev0, ev1])
                rec(fid, "corridor panel bounded at eval-end", ok,
                    f"panel {cor['panel_window'][0]}..{cor['panel_window'][1]}, last day ts "
                    f"{cor['panel_last_day_ts']} < {ev1}+1d; eval slice {cor['eval_window']}")

        # ---- manifest cross-checks: a train job may ONLY reference a train periods-file ----
        for kind in ("train", "eval"):
            try:  # FINDING 1: record, never raise out of the scan (see the fold loop)
                mpath = _resolve_root_artifact(fold_root, man["manifests"][kind],
                                               f"manifests.{kind}")
            except LeakError as e:
                rec("*", f"{kind}_jobs.manifest resolves INSIDE --fold-root", False,
                    str(e))
                continue
            if not os.path.exists(mpath):
                rec("*", f"{kind}_jobs.manifest exists", False, mpath)
                continue
            with open(mpath) as fh:
                lines = [ln.strip() for ln in fh
                         if ln.strip() and not ln.strip().startswith("#")]
            for ln in lines:
                job, dev, mem, cmd = ln.split(None, 3)
                fid = next((f["fold_id"] for f in man["folds"] if f"_{f['fold_id']}_" in job),
                           None)
                pf = _pf_from_cmd(cmd)
                if fid is None or pf is None:
                    rec("*", f"manifest line parse ({job})", False,
                        f"cannot map job to a fold / no --periods-file in: {ln[:120]}")
                    continue
                try:  # FINDING 1: record, never raise out of the scan
                    resolved_pf = _resolve_fold_artifact(
                        fold_root, fid, pf, f"{kind} job {job} --periods-file")
                except LeakError as e:
                    rec(fid, f"{kind} job {job} --periods-file resolves INSIDE "
                             f"--fold-root", False, str(e))
                    continue
                if not os.path.exists(resolved_pf):
                    rec(fid, f"{kind} job {job} periods-file exists", False, pf)
                    continue
                p0, p1, pf_assets = _read_periods(resolved_pf)
                fdef = next(f for f in man["folds"] if f["fold_id"] == fid)
                if kind == "train":
                    member = pf in fold_train_files.get(fid, set())
                    dates_ok = (_max_end_epoch(p1)
                                < _min_start_epoch(fdef["eval_window"][0]))
                    rec(fid, f"train job {job} references a fold TRAIN periods-file",
                        member and dates_ok,
                        f"--periods-file {pf}: membership in the fold's train files="
                        f"{member}; parsed window {p0}..{p1} ends before eval start "
                        f"{fdef['eval_window'][0]}={dates_ok}")
                else:
                    # MAJOR-2 fix: window equality alone let ANY same-window file through
                    # (incl. period_eval_leaky.json — the survivorship-leaky universe would
                    # silently become the graded unseen-years number). Require file IDENTITY
                    # with the fold's eval_pit AND assets == the PIT universe.
                    exp_pf = fdef["files"]["eval_pit"]
                    ident = pf == exp_pf
                    win_ok = [p0, p1] == fdef["eval_window"]
                    assets_ok = (fid in fold_uni_syms
                                 and pf_assets == fold_uni_syms[fid])
                    rec(fid, f"eval job {job} references the fold's eval_pit file",
                        ident and win_ok and assets_ok,
                        f"--periods-file {pf}: identity==files.eval_pit ({exp_pf})={ident}; "
                        f"window {p0}..{p1}=={fdef['eval_window']}={win_ok}; "
                        f"assets==universe_pit symbols={assets_ok}")
    finally:
        con.close()

    if fails:
        raise LeakError("LEAK ASSERT FAILED — fold artifacts REJECTED:\n  "
                        + "\n  ".join(fails))
    print(f"[837] validate: ALL {len(checks)} checks PASS ({_rel(fold_root)})")
    return checks


# =============================================================================
# fold corridor — 830 machinery on the eval window (padded panel, eval-slice numbers)
# =============================================================================

def _load_corridor830():
    """Import the TASK_830 corridor module by path (the 832 importlib pattern — dir name
    '830_corridor' is not an identifier). Reused pieces: load_panel (audited daily panel +
    819 liquidity filter), book_target_signs (819 book via trader.strategy), replay_floor
    (the in-env floor walk), oracle_positions (path_solver ceiling instrument), FEE/SLIP
    constants. NEVER modified."""
    spec = importlib.util.spec_from_file_location("corridor830", CORRIDOR_830)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fold_corridor(db_path, fold, pad_days=PAD_DAYS, position_size_pct=None, c830=None):
    """Per-eval-window corridor {floor_inenv, reference, ceiling} per cost tier.

    Panel = [eval_start - pad_days, eval_end] via 830 ``load_panel`` (UNMODIFIED defaults:
    include_dead, zf_max=0.15, min_active=365 — the pad exists exactly because the raw
    eval windows are shorter than that filter tolerates; 831-EXT crash precedent). The pad
    is PAST data only; the panel NEVER extends beyond eval_end (asserted, and re-checked
    from the emitted artifact by validate_folds). Every reported number is computed on the
    eval-window decision slice only:
      floor_inenv — 830 replay_floor over the panel (the env book enters the eval window
        already positioned — a live floor strategy would too); per-decision rets index k
        maps to row warm+k (the 830 _regime_slices convention); slice = rows with
        day_ts in [eval_start, eval_end) (backtest half-open window convention).
      reference   — backtest.window_metrics (gate-2 scorer: frozen 819 positions on the
        WHOLE panel incl. warmup pre-history, metrics only on the slice).
      ceiling     — 830 oracle_positions (path_solver = measurement instrument ONLY, H5)
        scored by backtest.regime_series on [r_lo, r_hi) — identical audited accounting.
    Annualization everywhere = backtest._sharpe(., period=H_rebal*K=1440)."""
    if c830 is None:
        c830 = _load_corridor830()
    fid = fold["fold_id"]
    e0, e1 = fold["eval"]
    pan_start = _date_add(e0, -pad_days)
    day_ts, close, logc, syms, info = c830.load_panel(db_path, pan_start, e1)
    A = close.shape[1]
    if position_size_pct is None:
        position_size_pct = 0.5 / A                       # the 830 default sizing
    s_ts, e_ts = utc_epoch(e0), utc_epoch(e1)
    if int(day_ts[-1]) >= utc_epoch(_date_add(e1, 1)):    # leak-side guard, fail loud
        raise LeakError(f"{fid}: corridor panel extends past eval end {e1} "
                        f"(last day {int(day_ts[-1])})")

    tgt, pres, warm = c830.book_target_signs(logc)
    r_lo = int(np.searchsorted(day_ts, s_ts))
    r_hi = int(np.searchsorted(day_ts, e_ts))
    H = trader_config.STRATEGY.H_rebal
    period = H * trader_config.K

    # reference — gate-2 scorer on the eval slice (positions use the padded pre-history)
    ref = backtest.window_metrics(logc, day_ts, start_ts=s_ts, end_ts=e_ts)
    if ref is None or ref["n"] < MIN_EVAL_DECISIONS:
        raise LeakError(f"{fid}: eval window {e0}..{e1} too thin for the corridor "
                        f"(window_metrics n={None if ref is None else ref['n']})")

    # ceiling — oracle book on the identical slice, identical accounting
    pos_orc, orc_alpha = c830.oracle_positions(logc, pres)
    res = backtest.regime_series(logc, np.asarray(pos_orc), pres, r_lo, r_hi, H, warm)
    if res is None:
        raise LeakError(f"{fid}: oracle slice empty on {e0}..{e1}")
    og, ot, _ = res
    ceiling = {c: _f(backtest._sharpe(backtest.net_series(og, ot, c), period))
               for c in GATE_COSTS}

    # floor — 830 replay through step_vector, rets sliced to eval rows
    floors, floor_diag = {}, {}
    n_eval = None
    for c in GATE_COSTS:
        fee, slip = (0.0, 0.0) if c == 0 else (c830.FEE_BPS, c830.SLIP_BPS)
        rep = c830.replay_floor(close, tgt, warm, fee, slip, position_size_pct)
        dec_ts = day_ts[warm:warm + rep["n"]]
        sel = (dec_ts >= s_ts) & (dec_ts < e_ts)
        n_eval = int(sel.sum())
        if n_eval < MIN_EVAL_DECISIONS:
            raise LeakError(f"{fid}: only {n_eval} in-env eval decisions on {e0}..{e1}")
        floors[c] = _f(c830._sharpe(rep["rets"][sel]))
        floor_diag[c] = {"final_pv_panel": rep["final_pv"],
                         "bankrupt_at": rep["bankrupt_at"]}

    out = {
        "task": "TASK_837", "fold_id": fid, "eval_window": [e0, e1],
        "panel_window": [pan_start, e1], "pad_days": pad_days,
        "panel_last_day_ts": int(day_ts[-1]),
        "db": _rel(db_path), "n_assets": A, "n_days": int(day_ts.size), "warm": warm,
        "n_eval_decisions_env": n_eval, "n_eval_decisions_engine": int(ref["n"]),
        "position_size_pct": position_size_pct,
        "oracle_gross_alpha_panel": _f(orc_alpha),
        "per_tier": {str(c): {"floor_inenv": floors[c],
                              "reference": _f(ref["gross"] if c == 0 else ref[f"net{c}"]),
                              "ceiling": ceiling[c]} for c in GATE_COSTS},
        "bh_eval": _f(ref["bh"]),
        "floor_diag": {str(c): floor_diag[c] for c in GATE_COSTS},
        "provenance": ("830 corridor machinery (importlib, unmodified) on a "
                       f"{pad_days}d-back-padded panel; ALL numbers are eval-slice only; "
                       "universe = 830-pattern liquidity-filtered daily panel (NOT the "
                       "fold's 64) — same yardstick caveat as 835 SPEC #10. "
                       "Daily source: fixture daily == db daily (LEDGER 830/E0-REAL)."),
        # M6 (verify-panel): the exact edge mismatch vs the 835 fold scorecard, with sign.
        "slice_convention": (
            "corridor slice = backtest half-open [eval_start, eval_end) on UTC-midnight "
            "day_ts; the daily panel load is end-EXCLUSIVE, so the last graded corridor "
            "decision is eval_end-2d (the final panel row only labels the previous "
            "decision). The 835 fold scorecard's minute loader is end-INCLUSIVE at LOCAL "
            "midnight and consumes window_size (60) rows of the eval panel for obs "
            "warmup: its graded decisions run ~eval_start+60d .. ~eval_end-1d. SIGN of "
            "the mismatch: the corridor covers ~60 MORE daily decisions at the LEFT edge "
            "and ~1 FEWER at the RIGHT edge (tz skew <=1d). Neither instrument ever "
            "touches data past eval_end. Documented (SPEC §corridor), not tuned."),
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    print(f"[837] corridor {fid} ({e0}..{e1}, {A} assets, {n_eval} eval decisions): "
          f"net@21 [{out['per_tier']['21']['floor_inenv']}, "
          f"{out['per_tier']['21']['ceiling']}] ref {out['per_tier']['21']['reference']}")
    return out


def corridor_for_root(db_path, fold_root, folds, pad_days=PAD_DAYS):
    c830 = _load_corridor830()
    outs = []
    for fold in folds:
        cor = fold_corridor(db_path, fold, pad_days=pad_days, c830=c830)
        outs.append(_write_json(os.path.join(fold_root, fold["fold_id"],
                                             "corridor.json"), cor))
    return outs


# =============================================================================
# aggregation — per-fold scorecards + corridors -> walk-forward summary + LEDGER row
# =============================================================================

def _fmt(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:+.3f}"


def aggregate(fold_root, scorecard_map, out_path, tag):
    """Fold scorecards (835 result JSONs — real runs or smoke stubs) graded against each
    fold's OWN corridor. Per-fold corridor positions ALWAYS shown; the pooled block never
    exists without them (spec: no pooled Sharpe without the per-fold view). Partial
    rollouts withhold corridor booleans (the 835 completeness-guard convention)."""
    man_path = os.path.join(fold_root, "fold_manifest.json")
    if not os.path.lexists(man_path):
        # TASK_904A MAJOR fix: mirror validate_folds' pre-check. Without it a missing
        # manifest raised a raw FileNotFoundError traceback instead of a parseable
        # `LEAK GATE:` refusal — indistinguishable from a crash to an orchestrator.
        raise LeakError(f"no fold_manifest.json under {_rel(fold_root)}")
    # F3: the SHARED guarded reader — see validate_folds' identical guard.
    man = _load_json_regular_file(man_path, LeakError,
                                  f"aggregate: {_rel(man_path)}")
    rows = []
    for f in man["folds"]:
        fid = f["fold_id"]
        cpath = os.path.join(fold_root, fid, "corridor.json")
        if not os.path.exists(cpath):
            raise LeakError(f"aggregate: no corridor.json for {fid} — run `corridor` "
                            f"first (a fold is never graded without its own corridor)")
        with open(cpath) as fh:
            cor = json.load(fh)
        sc_path = scorecard_map.get(fid)
        if sc_path is None or not os.path.exists(sc_path):
            raise LeakError(f"aggregate: no scorecard for {fid} (got {sc_path})")
        with open(sc_path) as fh:
            sc = json.load(fh)
        t21, t0 = sc["tiers"]["21"], sc["tiers"]["0"]
        partial = bool(t21.get("partial") or t0.get("partial"))
        net21, gross = _f(t21.get("sharpe_full")), _f(t0.get("sharpe_full"))
        fl, rf, ce = (cor["per_tier"]["21"][k] for k in
                      ("floor_inenv", "reference", "ceiling"))
        span = None if (fl is None or ce is None) else max(ce - fl, 1e-10)
        pct_hold = t21.get("pct_hold_agg")
        rows.append({
            "fold_id": fid,
            "train_window": f["train_window"], "eval_window": f["eval_window"],
            "scorecard": {"path": _rel(sc_path),
                          "checkpoint_label": sc.get("checkpoint_label"),
                          "stub": bool(sc.get("stub", False)),
                          "net21": net21, "gross": gross,
                          "pct_hold_agg_net21": pct_hold, "partial": partial,
                          "final_pv_net21": t21.get("final_pv")},
            "corridor": {"path": _rel(cpath), "per_tier": cor["per_tier"],
                         "n_eval_decisions_env": cor["n_eval_decisions_env"]},
            "position_in_corridor_net21": (None if partial or net21 is None or span is None
                                           else float((net21 - fl) / span)),
            "beats_floor_net21": (None if partial or net21 is None or fl is None
                                  else bool(net21 >= fl)),
            "beats_reference_net21": (None if partial or net21 is None or rf is None
                                      else bool(net21 >= rf)),
            "hold_collapse": (None if pct_hold is None
                              else bool(pct_hold > HOLD_COLLAPSE_PCT)),
        })

    # graded = non-partial folds with a number — the SAME set feeds mean/min/worst
    # (a partial fold must not name worst_fold while being excluded from the stats).
    graded = [r for r in rows
              if r["scorecard"]["net21"] is not None and not r["scorecard"]["partial"]]
    valid = [r["scorecard"]["net21"] for r in graded]
    pooled = {
        "n_folds": len(rows), "n_folds_graded": len(valid),
        "mean_net21": _f(np.mean(valid)) if valid else None,
        "min_net21": _f(np.min(valid)) if valid else None,
        "worst_graded_fold": (min(graded, key=lambda r: r["scorecard"]["net21"])
                              ["fold_id"] if graded else None),
        "all_beat_floor": (all(r["beats_floor_net21"] for r in rows)
                           if all(r["beats_floor_net21"] is not None for r in rows)
                           else None),
        "all_beat_reference": (all(r["beats_reference_net21"] for r in rows)
                               if all(r["beats_reference_net21"] is not None for r in rows)
                               else None),
        "note": "pooled view — NEVER read without the per-fold rows above "
                "(TASK_837: no pooled Sharpe without the per-fold view)",
    }
    is_stub = any(r["scorecard"]["stub"] for r in rows)
    per_fold_txt = " · ".join(
        f"{r['fold_id']} net@21 {_fmt(r['scorecard']['net21'])} in "
        f"[{_fmt(r['corridor']['per_tier']['21']['floor_inenv'])}, "
        f"{_fmt(r['corridor']['per_tier']['21']['ceiling'])}] "
        f"(ref {_fmt(r['corridor']['per_tier']['21']['reference'])}, "
        f"pos {_fmt(r['position_in_corridor_net21'])})"
        for r in rows)
    stub_note = (" FIXTURE-SMOKE with STUB scorecards — plumbing proof only, no verdict."
                 if is_stub else
                 " Scheduled-run numbers; R4 graduation verdict applied by the planner.")
    ledger_row = (
        f"| 837/R4 | {date.today().isoformat()} | H1 (learned) | Walk-forward ({tag}): "
        f"per-fold eval vs each window's OWN corridor "
        f"({' + '.join(r['fold_id'] for r in rows)}) | {per_fold_txt}; pooled mean "
        f"{_fmt(pooled['mean_net21'])} min {_fmt(pooled['min_net21'])} over "
        f"{pooled['n_folds_graded']}/{pooled['n_folds']} folds.{stub_note} | "
        f"per-fold positions above (never pooled alone) | {_rel(out_path)} |")

    summary = {"task": "TASK_837", "tag": tag, "fold_set": man["fold_set"],
               "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "folds": rows, "pooled": pooled, "ledger_row": ledger_row}
    _write_json(out_path, summary)

    print("=" * 90)
    print(f"WALK-FORWARD SUMMARY ({tag}) — per-fold FIRST (house rule)")
    print(f"{'fold':>12} | {'net@21':>8} {'gross':>8} | {'floor':>8} {'ref':>8} "
          f"{'ceil':>8} | {'pos':>6} {'>=floor':>7} {'>=ref':>5} {'collapse':>8}")
    for r in rows:
        pt = r["corridor"]["per_tier"]["21"]
        flag = {True: "YES", False: "no", None: "n/a"}
        print(f"{r['fold_id']:>12} | {_fmt(r['scorecard']['net21']):>8} "
              f"{_fmt(r['scorecard']['gross']):>8} | {_fmt(pt['floor_inenv']):>8} "
              f"{_fmt(pt['reference']):>8} {_fmt(pt['ceiling']):>8} | "
              f"{_fmt(r['position_in_corridor_net21']):>6} "
              f"{flag[r['beats_floor_net21']]:>7} {flag[r['beats_reference_net21']]:>5} "
              f"{flag[r['hold_collapse']]:>8}")
    print(f"pooled: mean net@21 {_fmt(pooled['mean_net21'])} min {_fmt(pooled['min_net21'])} "
          f"({pooled['n_folds_graded']}/{pooled['n_folds']} folds) — {pooled['note']}")
    print("LEDGER row:")
    print(ledger_row)
    print("=" * 90)
    return summary


# =============================================================================
# fixture smoke — end-to-end + the NEGATIVE leak tests (teeth)
# =============================================================================

def _stub_scorecard(path, fid, net21, gross, pct_hold):
    """Tiny 835-schema stub (the keys aggregate consumes) — smoke plumbing only."""
    def tier(sh):
        return {"sharpe_full": sh, "final_pv": 100000.0 * (1 + sh / 100),
                "pct_hold_agg": pct_hold, "partial": False,
                "steps_completed": 20, "steps_expected": 20,
                "avg_long": 1.0, "avg_short": 1.0, "bankrupt_at": -1}
    return _write_json(path, {
        "tag": f"837-smoke-stub-{fid}", "stub": True,
        "checkpoint_label": "STUB (no model — smoke aggregation plumbing)",
        "tiers": {"21": tier(net21), "0": tier(gross)}})


def _clone_fold_tree(smoke_root, case):
    """Copy the fold tree for a control/negative run. MAJOR-3 (original fix, now
    superseded): the job manifests were left pointing at the original tree because
    ``validate_folds`` resolved every path from this script's ``_REPO``, so even a
    PRISTINE copy was rejected on paths and the negative tests had zero discriminating
    power — the fix at the time was to re-point every repo-relative path inside the
    clone with an ``old_rel``->``new_rel`` text replace.

    TASK_905: that text-replace was this exact defect's other face — the CLI's
    ``validate --fold-root <copy>`` had NO such rewrite and silently re-read the
    ORIGINAL tree on any plain ``cp -r`` copy. The fix now lives once, in
    ``validate_folds`` itself (``_resolve_fold_artifact`` / ``_resolve_root_artifact``):
    every artifact is resolved against the ``fold_root`` it is actually pointed at, so
    a clone validates correctly with its manifest content UNCHANGED — the rewrite this
    function used to do is no longer needed for either caller. CLI and smoke helper now
    share the one implementation (870 remainder, scoped to this surface)."""
    neg_root = os.path.join(smoke_root, "_negative", case)
    if os.path.exists(neg_root):
        shutil.rmtree(neg_root)
    shutil.copytree(smoke_root, neg_root, ignore=shutil.ignore_patterns("_negative"))
    return neg_root


def _noop_control(smoke_root, db_path):
    """CONTROL (MAJOR-3): an UNMUTATED clone must validate CLEAN — otherwise every
    'rejection' below is a path confound, not a leak gate, and the smoke FAILS."""
    root = _clone_fold_tree(smoke_root, "control_noop")
    try:
        checks = validate_folds(root, db_path, verbose=False)
    except LeakError as e:
        raise RuntimeError(
            f"NEGATIVE-TEST CONTROL FAILED: the UNMUTATED clone was rejected — the "
            f"negative tests are path-confounded, not gate-driven:\n{e}")
    print(f"[837][negative:control_noop] CLEAN as required ({len(checks)} checks PASS "
          f"on the unmutated clone)")
    return {"case": "control_noop", "clean": True, "n_checks": len(checks)}


def _corrupt_and_expect_reject(smoke_root, db_path, case, mutate, expect):
    """Clone the validated smoke fold tree, apply ``mutate``, and REQUIRE validate to
    (a) REJECT it AND (b) NAME the targeted gate (``expect`` substring) in the
    rejection. A silent pass OR a rejection by the wrong check is itself a failure —
    a refactor that de-tooths the targeted gate must break the smoke."""
    neg_root = _clone_fold_tree(smoke_root, case)
    mutate(neg_root)
    try:
        validate_folds(neg_root, db_path, verbose=False)
    except LeakError as e:
        msg = str(e)
        if expect not in msg:
            raise RuntimeError(
                f"NEGATIVE TEST '{case}': REJECTED but NOT by the targeted gate "
                f"'{expect}' — the rejection reads:\n{msg}")
        line = next((l.strip() for l in msg.splitlines() if expect in l), expect)
        print(f"[837][negative:{case}] REJECTED by targeted gate '{expect}' — {line}")
        return {"case": case, "rejected": True, "targeted_check": expect,
                "named_in_rejection": True, "message": line}
    raise RuntimeError(f"NEGATIVE TEST FAILED: corrupted fold '{case}' was ACCEPTED — "
                       f"the leak gates have no teeth")


def _neg_train_into_eval(root):
    """Corruption A: the train periods-file end_date is pushed INTO the eval window."""
    p = os.path.join(root, "smokefold1", "period_train.json")
    with open(p) as fh:
        d = json.load(fh)
    d["periods"][0]["end_date"] = "2025-11-15"      # 15 days inside the eval window
    with open(p, "w") as fh:
        json.dump(d, fh, indent=2)


def _neg_tampered_universe(root):
    """Corruption B: universe_pit.json symbol list tampered (rank-1 symbol replaced) —
    the re-derivation equality must reject it."""
    p = os.path.join(root, "smokefold2", "universe_pit.json")
    with open(p) as fh:
        d = json.load(fh)
    d["symbols"][0] = "NOT_A_PIT_MEMBER/USD"
    with open(p, "w") as fh:
        json.dump(d, fh, indent=2)


def _neg_train_job_on_eval_file(root):
    """Corruption C: a train manifest job re-pointed at the fold's EVAL periods-file."""
    p = os.path.join(root, "train_jobs.manifest")
    with open(p) as fh:
        txt = fh.read()
    with open(p, "w") as fh:
        fh.write(txt.replace("period_train.json", "period_eval.json", 1))


def _neg_consistent_retamper(root):
    """Corruption D (MAJOR-1 teeth): an INTERNALLY-CONSISTENT tamper — the universe is
    truncated to 3 with the rule block, counts AND both pit periods-files edited to
    match. Every self-consistency check passes; only the pre-registration PIN
    (POLICY-PIN universe rule + the PINNED-params re-derivation) can reject it."""
    fdir = os.path.join(root, "smokefold1")
    up = os.path.join(fdir, "universe_pit.json")
    with open(up) as fh:
        u = json.load(fh)
    syms = u["symbols"][:3]
    u["symbols"], u["asset_ids"] = syms, u["asset_ids"][:3]
    u["counts"] = {s: u["counts"][s] for s in syms}
    u["n_selected"] = 3
    u["rule"]["n_assets_cap"] = 3
    with open(up, "w") as fh:
        json.dump(u, fh, indent=2)
    for name in ("period_train.json", "period_eval.json"):
        p = os.path.join(fdir, name)
        with open(p) as fh:
            d = json.load(fh)
        d["periods"][0]["assets"] = syms
        with open(p, "w") as fh:
            json.dump(d, fh, indent=2)


def _neg_eval_job_on_leaky(root):
    """Corruption E (MAJOR-2 teeth): an eval manifest job re-pointed at the SAME-WINDOW
    period_eval_leaky.json — window equality alone would let the survivorship-leaky
    universe silently become the graded unseen-years number; the eval-file IDENTITY
    gate must reject it."""
    p = os.path.join(root, "eval_jobs.manifest")
    with open(p) as fh:
        txt = fh.read()
    with open(p, "w") as fh:
        fh.write(txt.replace("period_eval.json", "period_eval_leaky.json", 1))


# =============================================================================
# TASK_904A — the smoke overwrite landmine: real rails are OUT-ROOTED to a scratch temp
# dir and DIFFED against the tracked tree, never written in place.
# =============================================================================

_SCRATCH_PREFIX = "real_smoke_scratch_"    # matches .gitignore's real_smoke_scratch_*/
_SCRATCH_STALE_SECS = 6 * 3600


def _newest_mtime(root):
    """The NEWEST mtime anywhere in ``root``'s tree — the age an in-flight run actually
    has. TASK_904A (cycle 3 / F4): ageing a scratch out-root by the TOP-LEVEL directory's
    own mtime measured the wrong thing. Writing into a SUBdirectory does not bump the
    parent's mtime, and a real-fold smoke run's tail (validate/diff, read-only) never
    touches the out-root again — so a tail phase running past the staleness threshold
    made a LIVE peer's working directory look like crash debris and eligible for rmtree.

    ``os.walk`` does not follow symlinked directories (followlinks=False default) and
    every entry is ``lstat``-ed, so a link inside the tree contributes its OWN mtime and
    is never chased out of the tree. Unreadable entries are skipped: this feeds a
    best-effort sweep, and a stat failure must never abort it."""
    newest = os.lstat(root).st_mtime
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            try:
                newest = max(newest, os.lstat(os.path.join(dirpath, name)).st_mtime)
            except OSError:
                continue
    return newest


def _sweep_stale_scratch(folds_dir, stale_secs=_SCRATCH_STALE_SECS):
    """Reap scratch out-roots left behind by CRASHED runs. Per-run unique names (mkdtemp)
    fixed the concurrent-run race that a single fixed scratch path had, but they also
    removed its self-healing property: nothing reaped them, so a SIGKILLed run left
    ``folds/real_smoke_scratch_<suffix>/`` behind forever.

    Deliberately narrow, and AGE-GATED so the race fix survives:
      * only DIRECT children of ``folds_dir`` whose basename starts with the gitignored
        prefix — a name that cannot collide with ``folds/real`` or ``folds/smoke``;
      * only real directories, never symlinks — checked on the ``folds_dir`` ROOT as well
        as on each child (cycle 3 / F5: a symlinked ``folds`` root is followed by
        ``isdir``/``listdir``, and an aged same-named subdir in the link's target would
        then be rmtree'd), so the sweep can never follow a link out of the gitignored
        pattern;
      * only entries whose whole TREE has been untouched for ``stale_secs`` (default 6h),
        aged on the newest mtime found ANYWHERE inside (cycle 3 / F4 — see
        ``_newest_mtime``: the top-level dir's own mtime does not move when a run writes
        into a subdirectory, so it under-reports a live peer's age). A smoke run takes ~2
        minutes, so a CONCURRENT run's scratch dir is never in scope;
      * best-effort: a failure to remove debris WARNs and continues. Debris must never
        be able to fail a run — each run materializes into its OWN fresh dir, so leftover
        dirs are already inert by construction (they are never read, never diffed, and
        never promoted into the tracked tree).
    Returns the list of removed (repo-relative) paths."""
    removed = []
    try:
        root_st = os.lstat(folds_dir)          # F5: lstat, never isdir — a SYMLINKED
    except OSError:                            # folds root would otherwise be followed
        return removed
    if not stat.S_ISDIR(root_st.st_mode):
        print(f"[837] WARN: not sweeping {_rel(folds_dir)} — it is not a real directory "
              f"(mode {stat.filemode(root_st.st_mode)})")
        return removed
    now = time.time()
    for name in sorted(os.listdir(folds_dir)):
        if not name.startswith(_SCRATCH_PREFIX):
            continue
        p = os.path.join(folds_dir, name)
        try:
            st = os.lstat(p)
            if not stat.S_ISDIR(st.st_mode):     # symlink/file/FIFO: not ours, skip
                continue
            # F4: age on the NEWEST mtime in the whole tree, not the top-level dir's own.
            if now - _newest_mtime(p) < stale_secs:   # an in-flight concurrent run
                continue
            shutil.rmtree(p)
            removed.append(_rel(p))
        except OSError as e:
            print(f"[837] WARN: could not sweep stale scratch out-root {_rel(p)}: {e}")
    if removed:
        print(f"[837] swept {len(removed)} stale scratch out-root(s) left behind by "
              f"crashed run(s): {removed}")
    return removed

# M4 identity contract (SPEC.md:61-70): universe_pit.json fields compared, EXCLUDING the
# volatile ones (generated/db/minute_coverage_check — a literal byte-identity contract
# on those would be unsatisfiable across runs/dbs by definition).
_UNIVERSE_CONTRACT_FIELDS = ("as_of", "as_of_ts_utc", "rule", "n_qualifying",
                             "n_selected", "symbols", "asset_ids", "counts")
_PERIOD_FILES = ("period_train.json", "period_eval.json",
                 "period_train_leaky.json", "period_eval_leaky.json")


def _diff_real_fold_tree(fresh_root, tracked_root, folds):
    """Diff a freshly (fixture-)materialized fold tree against the TRACKED
    ``folds/real/`` tree, per the M4 identity contract (SPEC.md:61-70): the four
    ``period_*.json`` per fold must be JSON-identical (they embed no volatile fields —
    the spec's 'byte-identical'), and ``universe_pit.json`` identical on the DERIVATION
    fields only. Read-only both sides. Returns a list of mismatch strings (empty ==
    identical == safe not to touch the tracked tree)."""
    problems = []
    for fold in folds:
        fid = fold["fold_id"]
        for name in _PERIOD_FILES:
            fp, tp = (os.path.join(fresh_root, fid, name),
                      os.path.join(tracked_root, fid, name))
            if not os.path.exists(tp):
                problems.append(f"{fid}/{name}: MISSING in tracked tree {_rel(tracked_root)}")
                continue
            with open(fp) as fh:
                fresh = json.load(fh)
            with open(tp) as fh:
                tracked = json.load(fh)
            if fresh != tracked:
                problems.append(f"{fid}/{name}: content differs from tracked")
        fp = os.path.join(fresh_root, fid, "universe_pit.json")
        tp = os.path.join(tracked_root, fid, "universe_pit.json")
        if not os.path.exists(tp):
            problems.append(f"{fid}/universe_pit.json: MISSING in tracked tree")
            continue
        with open(fp) as fh:
            fresh_u = json.load(fh)
        with open(tp) as fh:
            tracked_u = json.load(fh)
        for k in _UNIVERSE_CONTRACT_FIELDS:
            if fresh_u.get(k) != tracked_u.get(k):
                problems.append(f"{fid}/universe_pit.json[{k}]: differs "
                                f"(fresh={fresh_u.get(k)!r} tracked={tracked_u.get(k)!r})")
    return problems


def _expect_refusal(case, root, call):
    """Run ``call()`` expecting a ``MaterializeRefusal``, and require that ``root`` is
    byte-for-byte unchanged afterwards. A silent overwrite — or a refusal that mutated
    anything on its way out — is itself a failure."""
    before = _hash_tree(root)
    try:
        call()
    except MaterializeRefusal as e:
        after = _hash_tree(root)
        if after != before:
            raise RuntimeError(
                f"NEGATIVE TEST {case!r}: REFUSED but the content CHANGED anyway "
                f"({before} -> {after}) — the refusal is not fail-closed")
        print(f"[837][negative:{case}] REFUSED as required (content hash unchanged, "
              f"{before[:12]}...) — {e}")
        return {"case": case, "rejected": True, "hash_before": before,
                "hash_after": after, "content_untouched": True, "message": str(e)}
    raise RuntimeError(
        f"NEGATIVE TEST {case!r} FAILED: materialize() wrote into pre-existing content "
        f"instead of refusing — the overwrite gate has no teeth")


def _materialize_refusal_test(db):
    """TASK_904A gate (ii) battery for the overwrite gate. Three cases, run against
    throw-away temp trees:

      1. ``no_flag_over_existing`` — any non-empty out_root (here: a hand-placed decoy
         the tool never produced) is REFUSED without the explicit opt-in. This is the
         gate that has teeth, because the credential is the CALLER'S INTENT, not a value
         inside the directory — a forged manifest cannot buy write access.
      2. ``fixture_over_real_even_with_flag`` — a tree whose recorded provenance says
         REAL db is REFUSED when this call would replace it with fixture-derived
         content, even WITH the opt-in. This is the landmine itself.
      3. ``matching_provenance_with_flag`` — the ORDINARY case: re-materializing over a
         tree this tool produced from the same db, WITH the opt-in, PROCEEDS. Cycle 1's
         sentinel design failed exactly here (it locked out the SPEC.md-scheduled
         re-materialization forever), so the positive control ships alongside the
         negatives."""
    out = []

    # ---- 1. no flag over existing content -> refuse, content untouched ----
    decoy_root = tempfile.mkdtemp(prefix="837_decoy_")
    try:
        fid0 = REAL_FOLDS[0]["fold_id"]
        fdir = os.path.join(decoy_root, fid0)
        os.makedirs(fdir, exist_ok=True)
        # NOTE: the decoy deliberately carries a FORGED provenance block claiming this
        # tool produced it — under cycle 1's sentinel design that forgery bought write
        # access. Under the flag design it buys nothing.
        _write_json(os.path.join(decoy_root, "fold_manifest.json"), {
            "task": "TASK_837", "fold_set": "real",
            "universe_db": "fixtures/trading_fixture.db",
            _PROVENANCE_KEY: {"materializer": _MATERIALIZER_ID,
                              "universe_db": "fixtures/trading_fixture.db",
                              "db_kind": "fixture", "fold_set": "real",
                              "content_sha256": "0" * 64},
            "folds": [{"fold_id": f["fold_id"]} for f in REAL_FOLDS],
            "note": "hand-placed DECOY for TASK_904A's negative test — NOT produced by "
                    "materialize(), and carrying a FORGED provenance block."})
        _write_json(os.path.join(fdir, "period_train.json"),
                    {"DECOY": True, "periods": [{"name": "decoy"}]})
        out.append(_expect_refusal(
            "no_flag_over_existing", decoy_root,
            lambda: materialize(db, REAL_FOLDS, "real", decoy_root,
                                n_assets=PIT_N_ASSETS, min_days=PIT_MIN_DAYS,
                                jobs_db="trading_data.db")))
    finally:
        shutil.rmtree(decoy_root, ignore_errors=True)

    # ---- 1b. INTERNALLY INCONSISTENT provenance -> refuse, flag or no flag ----
    # (cycle 3 / F1) The decoy above forges a provenance block CONSISTENTLY. This one
    # lies in the single field the fixture-over-real refusal is computed from: it claims
    # db_kind "fixture" while naming a REAL universe_db. Reading that string verbatim
    # turned the one refusal no flag may override into an overridable one.
    liar_root = tempfile.mkdtemp(prefix="837_liar_")
    try:
        _write_json(os.path.join(liar_root, "fold_manifest.json"), {
            "task": "TASK_837", "fold_set": "real", "universe_db": "trading_data.db",
            _PROVENANCE_KEY: {"materializer": _MATERIALIZER_ID,
                              "universe_db": "trading_data.db",   # REAL db ...
                              "db_kind": "fixture",               # ... but claims fixture
                              "fold_set": "real", "content_sha256": "0" * 64},
            "note": "hand-edited/corrupt manifest whose provenance disagrees with itself."})
        _write_json(os.path.join(liar_root, "fold1", "period_train.json"), {"DECOY": True})
        out.append(_expect_refusal(
            "inconsistent_provenance_even_with_flag", liar_root,
            lambda: materialize(db, REAL_FOLDS, "real", liar_root,
                                n_assets=PIT_N_ASSETS, min_days=PIT_MIN_DAYS,
                                jobs_db="trading_data.db", allow_overwrite=True)))
    finally:
        shutil.rmtree(liar_root, ignore_errors=True)

    # ---- 2. fixture-over-real, WITH the flag -> still refuse ----
    real_prov_root = tempfile.mkdtemp(prefix="837_realprov_")
    try:
        _write_json(os.path.join(real_prov_root, "fold_manifest.json"), {
            "task": "TASK_837", "fold_set": "real", "universe_db": "trading_data.db",
            _PROVENANCE_KEY: {"materializer": _MATERIALIZER_ID,
                              "universe_db": "trading_data.db", "db_kind": "real",
                              "fold_set": "real", "generated": "2026-01-01T00:00:00+00:00",
                              "content_sha256": "0" * 64},
            "note": "stands in for a REAL-db-materialized fold tree (SPEC.md:61-70)."})
        out.append(_expect_refusal(
            "fixture_over_real_even_with_flag", real_prov_root,
            lambda: materialize(db, REAL_FOLDS, "real", real_prov_root,
                                n_assets=PIT_N_ASSETS, min_days=PIT_MIN_DAYS,
                                jobs_db="trading_data.db", allow_overwrite=True)))
    finally:
        shutil.rmtree(real_prov_root, ignore_errors=True)

    # ---- 3. POSITIVE control: matching provenance + flag -> proceeds ----
    # In-repo (gitignored) so validate_folds' repo-relative membership check applies —
    # see the note at the real-rails section below.
    folds_dir = os.path.join(EXP_DIR, "folds")
    os.makedirs(folds_dir, exist_ok=True)
    rerun_root = tempfile.mkdtemp(prefix="real_smoke_scratch_rerun_", dir=folds_dir)
    snapshot = rerun_root + "_snapshot"
    try:
        materialize(db, SMOKE_FOLDS, "smoke", rerun_root, n_assets=SMOKE_PIT_N_ASSETS,
                    min_days=PIT_MIN_DAYS, jobs_db=_rel(db))
        first = json.load(open(os.path.join(rerun_root, "fold_manifest.json")))
        shutil.copytree(rerun_root, snapshot)
        materialize(db, SMOKE_FOLDS, "smoke", rerun_root, n_assets=SMOKE_PIT_N_ASSETS,
                    min_days=PIT_MIN_DAYS, jobs_db=_rel(db), allow_overwrite=True)
        second = json.load(open(os.path.join(rerun_root, "fold_manifest.json")))
        if _PROVENANCE_KEY not in first or _PROVENANCE_KEY not in second:
            raise RuntimeError("POSITIVE CONTROL FAILED: materialize() did not record a "
                               f"{_PROVENANCE_KEY} block")
        # Compared on the M4 IDENTITY CONTRACT, not on the recorded content_sha256:
        # that hash covers the volatile ``generated`` stamps (universe_pit.json and the
        # job-manifest headers carry one), so two materializations landing on either side
        # of a second boundary hash differently while being the same folds. Cycle 2
        # compared the hashes and passed only when both calls fell inside one second —
        # a latent flake, and the wrong question besides: the contract this control
        # exists to prove is CONTENT identity (SPEC.md:61-70).
        diffs = _diff_real_fold_tree(rerun_root, snapshot, SMOKE_FOLDS)
        if diffs:
            raise RuntimeError(
                "POSITIVE CONTROL FAILED: an authorized re-materialization of the same "
                f"folds from the same db did not reproduce the same content: {diffs}")
        print("[837][positive:matching_provenance_with_flag] authorized "
              "re-materialization PROCEEDED (the SPEC.md path is NOT locked out)")
        out.append({"case": "matching_provenance_with_flag", "rejected": False,
                    "proceeded": True,
                    "content_identical_on_m4_contract": True,
                    "content_sha256": second[_PROVENANCE_KEY]["content_sha256"],
                    "message": "authorized re-materialization over a tool-produced tree "
                               "with matching provenance proceeds (B2 lockout fixed); "
                               "content identity checked on the M4 contract, since the "
                               "recorded content_sha256 covers volatile generated stamps"})
    finally:
        shutil.rmtree(rerun_root, ignore_errors=True)
        shutil.rmtree(snapshot, ignore_errors=True)
    return out


def run_smoke(args):
    """Gate (a): materialize + validate + corridor + stub-aggregate on the fixture,
    then the negative battery (no-op CONTROL must validate CLEAN + five corrupted
    clones each REJECTED by its targeted gate), AND the real-fold rails (materialize +
    validate [+ corridor unless --skip-real-corridor]). Writes
    results_fixture_smoke.json."""
    db = FIXTURE_DB
    result = {"task": "TASK_837", "smoke": True, "db": _rel(db),
              "generated": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    # ---- 0. reap scratch out-roots left behind by CRASHED earlier runs (age-gated, so
    # a concurrent run's dir is never touched — see _sweep_stale_scratch) ----
    result["swept_stale_scratch"] = _sweep_stale_scratch(os.path.join(EXP_DIR, "folds"))

    # ---- 1. smoke folds: materialize -> validate (leak gates) ----
    smoke_root = os.path.join(EXP_DIR, "folds", "smoke")
    if os.path.exists(smoke_root):
        shutil.rmtree(smoke_root)
    materialize(db, SMOKE_FOLDS, "smoke", smoke_root,
                n_assets=SMOKE_PIT_N_ASSETS, min_days=PIT_MIN_DAYS, jobs_db=_rel(db))
    checks = validate_folds(smoke_root, db)
    result["smoke_folds"] = {"root": _rel(smoke_root), "n_checks": len(checks),
                             "all_pass": True}

    # ---- 2. per-fold corridors (830 machinery, eval-slice) ----
    corridor_for_root(db, smoke_root, SMOKE_FOLDS, pad_days=args.pad_days)
    checks2 = validate_folds(smoke_root, db, verbose=False)   # corridor bounds re-checked
    result["smoke_corridors"] = {
        f["fold_id"]: json.load(open(os.path.join(smoke_root, f["fold_id"],
                                                  "corridor.json")))["per_tier"]
        for f in SMOKE_FOLDS}
    result["smoke_folds"]["n_checks_with_corridor"] = len(checks2)

    # ---- 3. stub scorecards -> aggregation ----
    sc_map = {
        "smokefold1": _stub_scorecard(os.path.join(smoke_root, "smokefold1",
                                                   "scorecard_stub.json"),
                                      "smokefold1", net21=0.30, gross=0.55, pct_hold=95.0),
        "smokefold2": _stub_scorecard(os.path.join(smoke_root, "smokefold2",
                                                   "scorecard_stub.json"),
                                      "smokefold2", net21=-0.20, gross=0.05, pct_hold=99.5),
    }
    summary = aggregate(smoke_root, sc_map,
                        os.path.join(smoke_root, "walkforward_summary.json"),
                        tag="fixture-smoke-stub")
    result["smoke_aggregate"] = {"pooled": summary["pooled"],
                                 "ledger_row": summary["ledger_row"]}

    # ---- 4. NEGATIVE leak tests: no-op control CLEAN first (de-confound), then every
    # corrupted fold must be REJECTED *by its targeted gate* (named in the rejection) ----
    result["negative_control"] = _noop_control(smoke_root, db)
    result["negative_tests"] = [
        _corrupt_and_expect_reject(smoke_root, db, "train_into_eval",
                                   _neg_train_into_eval,
                                   "ASSERT-1"),
        _corrupt_and_expect_reject(smoke_root, db, "tampered_pit_universe",
                                   _neg_tampered_universe,
                                   "ASSERT-2 PIT universe re-derivation"),
        _corrupt_and_expect_reject(smoke_root, db, "train_job_on_eval_file",
                                   _neg_train_job_on_eval_file,
                                   "references a fold TRAIN periods-file"),
        _corrupt_and_expect_reject(smoke_root, db, "consistent_universe_retamper",
                                   _neg_consistent_retamper,
                                   "POLICY-PIN universe rule"),
        _corrupt_and_expect_reject(smoke_root, db, "eval_job_on_leaky_universe",
                                   _neg_eval_job_on_leaky,
                                   "references the fold's eval_pit file"),
    ]
    # TASK_904A: the overwrite gate itself — materialize() refuses to write into ANY
    # non-empty out_root without the caller's explicit opt-in (which smoke never gives),
    # refuses fixture-over-real even WITH it, and still permits the ordinary authorized
    # re-materialization. Matters for every OTHER caller, not just smoke.
    result["negative_materialize_refusal"] = _materialize_refusal_test(db)

    # ---- 5. the REAL fold rails (the default manifest deliverable) — TASK_904A: OUT-
    # ROOTED to a scratch dir. Smoke NEVER writes into the tracked folds/real/ tree (that
    # used to silently overwrite tracked files — the landmine this defuses). The fresh
    # (fixture-derived) materialization is DIFFED against the tracked tree per the M4
    # identity contract (SPEC.md:61-70); any mismatch is a smoke FAILURE (refuse), never
    # a silent overwrite of tracked content.
    # NOTE 1 (rewritten, TASK_905): the scratch dir lives INSIDE the repo
    # (folds/real_smoke_scratch_*/, gitignored) purely by convention now — it is no
    # longer load-bearing. It USED to be: validate_folds' membership check compared
    # ``os.path.relpath(p, _REPO)`` (recomputed from fold_manifest.json's recorded path)
    # against ``_rel(p)`` (written into the job manifests), and those two spellings agree
    # ONLY for in-repo paths (_rel() falls back to an absolute path outside the repo,
    # relpath does not), so an out-of-repo scratch dir produced a FALSE REJECTION —
    # "membership in the fold's train files=False" on every train job even though the
    # file was correct. TASK_905 removed that recomputation: membership now compares the
    # RECORDED strings on both sides (validate_folds, "Membership is about recorded
    # IDENTITY"), which is spelling-independent, so an out-of-repo scratch dir validates
    # correctly. Kept in-repo anyway for the gitignore/cleanup convention and because
    # _hash_tree/_diff_real_fold_tree comparisons below read easier under _rel().
    # NOTE 2: the name is UNIQUE per run (mkdtemp), not a fixed constant: this repo runs
    # parallel factory lanes routinely, and two concurrent `smoke` runs sharing one fixed
    # scratch path would race (one rmtree-ing the other's tree mid-validate). ----
    real_root = os.path.join(EXP_DIR, "folds", "real")
    folds_dir = os.path.join(EXP_DIR, "folds")
    os.makedirs(folds_dir, exist_ok=True)
    tmp_real_root = tempfile.mkdtemp(prefix="real_smoke_scratch_", dir=folds_dir)
    try:
        real_root_before_hash = (_hash_tree(real_root)
                                 if os.path.isdir(real_root) else None)
        materialize(db, REAL_FOLDS, "real", tmp_real_root,
                    n_assets=PIT_N_ASSETS, min_days=PIT_MIN_DAYS,
                    jobs_db="trading_data.db")
        if not args.skip_real_corridor:
            corridor_for_root(db, tmp_real_root, REAL_FOLDS, pad_days=args.pad_days)
        real_checks = validate_folds(tmp_real_root, db)

        mismatches = _diff_real_fold_tree(tmp_real_root, real_root, REAL_FOLDS)
        if mismatches:
            raise LeakError(
                "real-fold content mismatch: the fresh (fixture-derived) "
                f"materialization does not reproduce the TRACKED {_rel(real_root)} "
                "tree on the M4 identity contract fields (SPEC.md:61-70) — refusing to "
                f"overwrite tracked content; mismatches: {mismatches}")
        real_root_after_hash = (_hash_tree(real_root)
                                if os.path.isdir(real_root) else None)
        if real_root_after_hash != real_root_before_hash:
            raise RuntimeError(
                "INVARIANT VIOLATED: the tracked folds/real/ tree changed during smoke "
                f"despite the out-rooted rewrite ({real_root_before_hash} -> "
                f"{real_root_after_hash}) — smoke must NEVER write tracked FOLD "
                "content.")

        result["real_folds"] = {
            "root": _rel(real_root),
            "materialized_in": (f"{_rel(tmp_real_root)} — a unique per-run, gitignored "
                                "scratch dir, deleted at the end of the run; NEVER the "
                                "tracked tree"),
            "n_checks": len(real_checks), "all_pass": True,
            "tracked_content_verified_identical": True,
            "tracked_tree_hash_unchanged": True,
            "tracked_folds_real_sha256": real_root_after_hash,
            "corridors": ({f["fold_id"]: json.load(open(os.path.join(
                tmp_real_root, f["fold_id"], "corridor.json")))["per_tier"]
                for f in REAL_FOLDS} if not args.skip_real_corridor else "SKIPPED"),
            "universe_note": "PIT universes from the fixture's FULL ohlcv_daily panel "
                             "(fixture daily == db daily, LEDGER 830/E0-REAL); scheduled "
                             "re-materialization vs trading_data.db must reproduce them "
                             "content-identically (M4 contract: volatile fields excluded — "
                             "SPEC §PIT). Verified HERE by diff against the tracked tree, "
                             "not by writing to it (TASK_904A).",
        }
    finally:
        shutil.rmtree(tmp_real_root, ignore_errors=True)

    result["verdict"] = ("PASS — folds materialized with correct window boundaries; leak "
                         "asserts + POLICY pre-registration PIN enforced IN CODE (good "
                         "folds pass, the UNMUTATED control clone validates CLEAN, and "
                         "all 5 corrupted folds are REJECTED each BY ITS TARGETED gate); "
                         "materialize() refuses to write into any non-empty out-root "
                         "without an explicit opt-in, refuses fixture-over-real, and "
                         "refuses fixture-derived content onto the canonical tracked "
                         "real-fold tree by its identity (TASK_904A); corridors "
                         "eval-slice-bounded; aggregation emits per-fold-first summary "
                         "+ LEDGER row; real-fold rails verified by diff against the "
                         "tracked tree WITHOUT writing to it. Plumbing proof — no "
                         "training, no promotion claims.")
    # TASK_904A honest-claim narrowing: what this run actually measured about tracked
    # paths. NOT "zero writes to tracked paths" — results_fixture_smoke.json is itself
    # tracked and is rewritten by every smoke run, by design and since 2026-07-08.
    result["tracked_write_claim"] = (
        "MEASURED: the tracked experiments/837_walkforward/folds/real/ tree is "
        "byte-for-byte unchanged across this run (content hash A/B, see "
        "real_folds.tracked_folds_real_sha256), and the real rails were materialized "
        "into a per-run gitignored scratch dir instead. NOT CLAIMED: 'zero writes to "
        "tracked paths' — this artifact (experiments/837_walkforward/"
        "results_fixture_smoke.json) is tracked and IS written by every smoke run, "
        "which is its purpose.")
    out = os.path.join(EXP_DIR, "results_fixture_smoke.json")
    _write_json(out, result)
    print(f"\n[837] SMOKE PASS -> {_rel(out)}")
    return result


# =============================================================================
# CLI
# =============================================================================

def _fold_set(name):
    return {"real": REAL_FOLDS, "smoke": SMOKE_FOLDS}[name]


def main():
    ap = argparse.ArgumentParser(description="TASK_837 walk-forward split harness (R4)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("materialize", help="emit fold periods-files/universes/manifests")
    p.add_argument("--db", default=FIXTURE_DB,
                   help="db for PIT universe derivation (read-only; default fixture — "
                        "scheduled runs pass trading_data.db)")
    p.add_argument("--fold-set", choices=("real", "smoke"), default="real",
                   help="pre-registered fold set (default: the REAL folds)")
    p.add_argument("--out-root", default=None,
                   help="output root (default: <this dir>/folds/<fold-set>)")
    p.add_argument("--n-assets", type=int, default=None,
                   help="PIT cap (default 64 real / 8 smoke — pre-registered)")
    p.add_argument("--min-days", type=int, default=PIT_MIN_DAYS)
    p.add_argument("--jobs-db", default=None,
                   help="db path written into job manifests (default: trading_data.db "
                        "for real, the fixture for smoke)")
    p.add_argument("--overwrite-existing", action="store_true",
                   help="authorize writing into a NON-EMPTY --out-root (TASK_904A). "
                        "Without it, materialize refuses rather than overwrite an "
                        "existing fold tree — the gate is your explicit intent, not "
                        "anything stored inside the directory. It is NECESSARY but not "
                        "SUFFICIENT for the canonical tracked real-fold tree: writing "
                        "there additionally requires --fold-set real AND a real-classed "
                        "--db, all three agreeing, or the write is refused. Replacing any "
                        "other REAL-db-materialized tree with fixture-derived content is "
                        "refused outright, flag or no flag.")

    p = sub.add_parser("validate", help="run the leak gates on generated fold artifacts")
    p.add_argument("--db", default=FIXTURE_DB)
    p.add_argument("--fold-root", required=True)

    p = sub.add_parser("corridor", help="per-fold eval-window corridors (830 machinery)")
    p.add_argument("--db", default=FIXTURE_DB)
    p.add_argument("--fold-set", choices=("real", "smoke"), default="real")
    p.add_argument("--fold-root", default=None)
    p.add_argument("--pad-days", type=int, default=PAD_DAYS)
    p.add_argument("--overwrite-existing", action="store_true",
                   help="authorize writing corridor.json INTO the canonical tracked "
                        "real-fold tree (TASK_904A). Bare `corridor` used to rewrite "
                        "folds/real/fold*/corridor.json with no gate at all. As for "
                        "materialize, the unlock is a conjunction: --fold-set real AND a "
                        "real-classed --db AND this flag. Everywhere else this flag is a "
                        "no-op — only that one tree is fenced.")

    p = sub.add_parser("aggregate", help="fold scorecards + corridors -> summary")
    p.add_argument("--fold-root", required=True)
    p.add_argument("--scorecard", action="append", default=[],
                   metavar="FOLD_ID=PATH", help="835 result JSON per fold (repeatable)")
    p.add_argument("--out", default=None)
    p.add_argument("--tag", default="walkforward")

    p = sub.add_parser("smoke", help="fixture smoke: end-to-end + negative leak tests")
    p.add_argument("--pad-days", type=int, default=PAD_DAYS)
    p.add_argument("--skip-real-corridor", action="store_true",
                   help="skip the real-fold corridor computation (rails only)")

    args = ap.parse_args()
    try:
        if args.cmd == "materialize":
            out_root = args.out_root or os.path.join(EXP_DIR, "folds", args.fold_set)
            n_assets = args.n_assets or (PIT_N_ASSETS if args.fold_set == "real"
                                         else SMOKE_PIT_N_ASSETS)
            jobs_db = args.jobs_db or ("trading_data.db" if args.fold_set == "real"
                                       else _rel(args.db))
            materialize(args.db, _fold_set(args.fold_set), args.fold_set, out_root,
                        n_assets=n_assets, min_days=args.min_days, jobs_db=jobs_db,
                        allow_overwrite=args.overwrite_existing)
            validate_folds(out_root, args.db)
        elif args.cmd == "validate":
            validate_folds(args.fold_root, args.db)
        elif args.cmd == "corridor":
            root = args.fold_root or os.path.join(EXP_DIR, "folds", args.fold_set)
            # TASK_904A item 3 — the identity fence binds at the shared write choke point,
            # so `corridor` into the canonical tracked real tree needs the same
            # affirmative unlock `materialize` does. This is the gate whose ABSENCE moved
            # the tracked tree hash twice in two days (a bare `corridor` rewrote
            # folds/real/fold{1,2}/corridor.json with no gate whatsoever). Nothing about
            # the corridor WRITER changes here — that redesign is TASK_904B.
            with _canonical_write_scope(root, args.db, args.fold_set,
                                        args.overwrite_existing, who="corridor"):
                corridor_for_root(args.db, root, _fold_set(args.fold_set),
                                  pad_days=args.pad_days)
            validate_folds(root, args.db, verbose=False)
        elif args.cmd == "aggregate":
            sc = dict(kv.split("=", 1) for kv in args.scorecard)
            out = args.out or os.path.join(args.fold_root, "walkforward_summary.json")
            aggregate(args.fold_root, sc, out, tag=args.tag)
        elif args.cmd == "smoke":
            run_smoke(args)
    except LeakError as e:
        print("\n" + "!" * 78, file=sys.stderr)
        print(f"LEAK GATE: {e}", file=sys.stderr)
        print("!" * 78, file=sys.stderr)
        sys.exit(1)
    except MaterializeRefusal as e:
        print("\n" + "!" * 78, file=sys.stderr)
        print(f"MATERIALIZE REFUSED: {e}", file=sys.stderr)
        print("!" * 78, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
