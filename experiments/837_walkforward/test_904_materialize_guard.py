"""TASK_904A — mutation tests for walkforward.materialize's overwrite gate, the unified
guarded manifest reader, and the smoke out-rooting sweep.

Cycle 1 gated overwrites on a SENTINEL STRING stored in the very file it guarded. A
verification panel demonstrated that design failed in both directions at once:

  * FORGEABLE — the sentinel is a public constant bound to nothing, so pasting it into a
    hand-made directory bought write access to the exact tree it protected (three
    escapes: forged sentinel in a copied tree, hand-forged manifest, symlinked manifest);
  * LOCKED OUT — the tracked ``folds/real/`` tree (2026-07-08) predates the sentinel and
    could never carry it, so the SPEC.md-scheduled real-db re-materialization refused
    deterministically, with no documented recovery.

The remedy under test: the GATE is the caller's explicit opt-in (``allow_overwrite`` /
``--overwrite-existing``); recorded provenance is DISCLOSURE with the refusals below
attached to it — none of them flag-overridable.

CYCLE 3 additions (a fail-closed re-verify found the cycle-2 gate holed in several
places; only the ones in TASK_904A's scope are exercised here — the corridor/aggregate
WRITER family (F2 in the cycle-3 record) is TASK_904B, out of scope for this suite):
  F1  the fixture-over-real refusal read the manifest's OWN ``db_kind`` string, so a
      directory could lie in the one field that refusal is computed from. The class is
      now re-derived from ``universe_db``; a stored ``db_kind`` that disagrees is refused
      as corruption.
  F3  the FIFO/type guard was wired into one manifest reader of three; ``validate`` and
      ``aggregate`` still opened bare and hung. One shared guarded reader now.
  F4  the stale-scratch sweeper aged an out-root by the TOP-LEVEL dir's mtime, which a
      nested write does not bump — it could reap a LIVE peer. Ages on the newest mtime
      in the tree now.
  F5  the sweep trusted that the ``folds`` root itself was not a symlink.

RUN-42 CYCLE 1 (fixed here, mutation-proved below): ``_assert_out_root_writable``'s
fixture-over-real refusal RE-DERIVED its class from the TARGET DIRECTORY'S OWN
``fold_manifest.json`` — and the tracked golden ``folds/real/`` tree genuinely WAS
materialized from the fixture in July, so that re-derivation comes back "fixture" and
the refusal never fired for the one tree it exists to protect. Cycle 1's remedy was an
identity fence comparing ``realpath(out_root)`` against a ``__file__``-relative constant.

RUN-42 CYCLE 2 — that remedy was itself holed, twice, with ONE root cause: **the fence held
the CALLER'S WORD about the destination, anchored to an accident of invocation, instead of
the DESTINATION'S OWN IDENTITY.**

  BLOCKER 1  the anchor was ``__file__``-relative, so running the IDENTICAL script from a
    COPY of the directory (a /tmp reconstruction, a HEAD-control worktree A/B rig — both
    standing techniques in this repo) while pointing ``--out-root`` at a real tree by
    absolute path made the fence go INERT: no refusal, no warning, write proceeds. Cycle 1's
    own test hid this, because it proved BOTH directions by MONKEYPATCHING
    ``_CANONICAL_REAL_ROOT`` — treating the anchor's correctness as an axiom and never once
    exercising the derivation that was broken.
  BLOCKER 2  the predicate also required the caller to SELF-REPORT the destination's nature
    (``fold_set == "real" and this_kind == "fixture"``), so a caller who merely DISAGREED
    with the destination switched the guard off: ``materialize --fold-set smoke --out-root
    <canonical folds/real> --overwrite-existing`` wrote 8-asset smoke content into the
    tracked 64-asset real rails, past both refusals.

The fix under test is destination-first and fail-closed: the anchor is the canonical
real-fold path of the git checkout the DESTINATION lives in (matched by path and by
``(st_dev, st_ino)``, refusing loudly if unresolvable); the predicate is destination-only,
so caller disagreement is DISQUALIFYING rather than exculpatory; and it binds at the shared
write choke point, so ``corridor`` is covered too. Every test below exercises the REAL
derivation against a THROW-AWAY /tmp directory made to look like a checkout (``_fake_repo``)
— never the tracked tree, and never by monkeypatching the anchor.

RUN-42 MAJORs (fixed here):
  * a falsy ``--db`` made both refusals inert (``_db_kind`` returned a third value,
    ``"unknown"``, while both branches gated on ``== "fixture"``), surfacing as an
    incidental ``sqlite3``/``TypeError`` crash with no parseable refusal prefix — see
    ``test_falsy_db_*``;
  * ``validate_folds``/``aggregate`` did not type-check the shared reader's return, so
    valid JSON of the wrong type (``[1,2,3]`` / ``null`` / ``"hello"``) escaped as a raw
    ``AttributeError``/``TypeError`` traceback instead of a parseable refusal — see
    ``test_wrong_type_manifest_*``;
  * ``aggregate`` on a missing manifest raised a raw ``FileNotFoundError`` — see
    ``test_aggregate_refuses_missing_manifest_cleanly``.

===============================================================================
DISCLOSURE — the 7 tests from the cycle-3 patch that this suite DROPS as 904B
===============================================================================
The scoping decision stands (all 7 exercise the corridor/aggregate WRITER family, which is
TASK_904B). But the earlier claim that each one "ERRORS against this tree on missing 904B
symbols" is TRUE FOR ONLY 4 OF THE 7. Measured, per test:

  CLAIM HOLDS (4/7) — ``AttributeError: ... '_corridor_path'``, i.e. genuinely a missing
  904B symbol:
    * test_corridor_content_change_refused_without_flag_then_allowed
    * test_corridor_fixture_over_real_refused_even_with_flag
    * test_corridor_into_real_provenance_tree_refused_before_computing
    * test_corridor_fifo_refused_not_opened
  CLAIM IS FALSE (3/7) — these RUN TO COMPLETION against live code and FAIL on real
  behaviour, documenting real unguarded gaps rather than an absent symbol:
    * test_corridor_idempotent_recompute_writes_nothing  -> AssertionError:
      ``corridor_for_root`` wrote 2 files on an identical recompute. STILL TRUE after this
      task: idempotent recompute is 904B, not touched here.
    * test_bare_corridor_cli_leaves_tracked_real_tree_identical -> AssertionError: bare
      ``corridor`` MUTATED the tracked tree. **This one WOULD NOW PASS**: the identity fence
      binds at the shared write choke point, so a bare ``corridor`` into the canonical tree
      is refused (exit 1) before anything is written — see
      ``test_bare_corridor_into_canonical_tree_is_refused``, which is that test rewritten to
      run against a /tmp fake checkout instead of the tracked tree.
    * test_stub_summary_never_overwrites_a_real_summary -> DID NOT RAISE: ``aggregate``
      silently overwrote a "real" summary. STILL TRUE for any non-canonical destination
      (904B owns stub-vs-real detection); into the canonical tree ``aggregate`` is now
      refused outright by the choke point.

ACCOUNTING NOTE: this file does not exist at HEAD and was never committed, so its honest
baseline is **0 tests** and the delta is **0 -> 54, all new**. Any "N kept + M new" framing
is meaningful only relative to the uncommitted source patch, not to the repository.

Run (server, wrapped):
    tools/safe_run.sh --name t904a --mem 16G -- \
        python -m pytest experiments/837_walkforward/test_904_materialize_guard.py -q
"""
import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))


def _load_wf():
    spec = importlib.util.spec_from_file_location(
        "wf904a", os.path.join(_HERE, "walkforward.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


wf = _load_wf()
TRACKED_REAL = os.path.join(_HERE, "folds", "real")


def _fake_repo(base, name, populate=False):
    """A throw-away /tmp directory made to LOOK like a checkout of this repo: a real
    ``.git`` plus the canonical real-fold path. Under the destination-first rule its
    ``experiments/837_walkforward/folds/real`` therefore IS a canonical tracked real-fold
    tree — which is exactly what lets these tests exercise the REAL anchor derivation
    (never a monkeypatched one) without a single write ever being aimed at the repo's own
    tracked tree.

    ``populate=True`` copies the tracked tree's CONTENT in (read-only on our side), for the
    tests that need to prove content is left untouched — or, under mutation, clobbered.
    ``populate=False`` leaves ``folds/real`` EMPTY, which makes a refusal unambiguous: an
    empty out-root needs no flag under every other rule in the module, so only the identity
    fence can possibly refuse it.

    Returns ``(repo_root, canonical_real_dir)``."""
    root = os.path.join(base, name)
    subprocess.run(["git", "init", "-q", root], check=True, capture_output=True)
    canonical = os.path.join(root, "experiments", "837_walkforward", "folds", "real")
    if populate:
        os.makedirs(os.path.dirname(canonical), exist_ok=True)
        shutil.copytree(TRACKED_REAL, canonical)
    else:
        os.makedirs(canonical, exist_ok=True)
    return root, canonical


def _stand_in_real_db(path):
    """A db standing in for ``trading_data.db``: fixture-DERIVED (so it carries the tables
    ``materialize``/``corridor`` read) but deliberately NOT byte-identical to the project
    fixture.

    RUN-42 CYCLE 3: the authorizing db classifier now establishes fixture identity by
    CONTENT as well as by path — sha256 + size against the tracked pin
    ``fixtures/FIXTURE_IDENTITY.json`` — so a bare ``shutil.copyfile(FIXTURE_DB, ...)``
    under a real-sounding name is correctly classified FIXTURE and refused. A byte-identical
    copy of the fixture was always a fiction as a stand-in for a REAL database; these tests
    now use one that differs (an extra marker table), which is what makes them a positive
    control for "proven real" rather than an accident of the old path-only comparison."""
    shutil.copyfile(wf.FIXTURE_DB, path)
    con = sqlite3.connect(path)
    try:
        con.execute("CREATE TABLE IF NOT EXISTS _stand_in_for_real_db(marker TEXT)")
        con.execute("INSERT INTO _stand_in_for_real_db VALUES ('TASK_904A stand-in')")
        con.commit()
    finally:
        con.close()
    assert wf._db_class(path)[0] == wf._DB_REAL, "the stand-in must classify POSITIVELY real"
    return path


_FIXTURE_DEFAULT = object()   # a real sentinel: `db=None` is a VALUE these tests pass


def _materialize_real(out_root, db=_FIXTURE_DEFAULT, fold_set="real", **kw):
    folds = wf.REAL_FOLDS if fold_set == "real" else wf.SMOKE_FOLDS
    n = wf.PIT_N_ASSETS if fold_set == "real" else wf.SMOKE_PIT_N_ASSETS
    return wf.materialize(wf.FIXTURE_DB if db is _FIXTURE_DEFAULT else db,
                          folds, fold_set, out_root,
                          n_assets=n, min_days=wf.PIT_MIN_DAYS,
                          jobs_db="trading_data.db", **kw)


@pytest.fixture
def scratch():
    """A UNIQUE in-repo scratch root (mkdtemp), matching what run_smoke uses: in-repo so
    validate_folds' repo-relative membership check applies, unique so parallel runs of
    this suite cannot race each other."""
    folds_dir = os.path.join(_HERE, "folds")
    os.makedirs(folds_dir, exist_ok=True)
    d = tempfile.mkdtemp(prefix="real_smoke_scratch_t904a_", dir=folds_dir)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _materialize_smoke(root, **kw):
    return wf.materialize(wf.FIXTURE_DB, wf.SMOKE_FOLDS, "smoke", root,
                          n_assets=wf.SMOKE_PIT_N_ASSETS, min_days=wf.PIT_MIN_DAYS,
                          jobs_db=wf._rel(wf.FIXTURE_DB), **kw)


# ---------------------------------------------------------------- the gate itself ----

def test_empty_out_root_needs_no_flag(scratch):
    """An empty (or absent) out_root is the first materialization — no flag required."""
    m = _materialize_smoke(scratch)
    assert m["fold_set"] == "smoke"
    assert os.path.exists(os.path.join(scratch, "fold_manifest.json"))


def test_nonempty_out_root_refused_without_flag(scratch):
    _materialize_smoke(scratch)
    before = wf._hash_tree(scratch)
    with pytest.raises(wf.MaterializeRefusal) as ei:
        _materialize_smoke(scratch)
    assert wf._hash_tree(scratch) == before, "refusal must leave the tree untouched"
    msg = str(ei.value)
    assert "--overwrite-existing" in msg, "the refusal must name the authorizing flag"
    assert wf._rel(scratch) in msg, "the refusal must name the directory"


def test_authorized_rerun_proceeds_and_is_reproducible(scratch, tmp_path):
    """B2: the ordinary re-materialization must NOT be locked out, and must reproduce the
    same fold CONTENT.

    Compared on the M4 identity contract (SPEC.md:61-70), NOT on the recorded
    ``content_sha256``: that hash covers the volatile ``generated`` stamps in
    universe_pit.json and the job-manifest headers, so two materializations either side
    of a second boundary hash differently while being identical folds. Cycle 2 asserted
    hash equality here and passed only by finishing inside one second."""
    first = _materialize_smoke(scratch)
    snap = str(tmp_path / "snap")
    shutil.copytree(scratch, snap)
    second = _materialize_smoke(scratch, allow_overwrite=True)
    assert wf._PROVENANCE_KEY in first and wf._PROVENANCE_KEY in second
    assert wf._diff_real_fold_tree(scratch, snap, wf.SMOKE_FOLDS) == []


# --------------------------------------- B1: the three demonstrated sentinel forgeries ----

def test_b1_forged_provenance_buys_nothing(scratch):
    """Escape (i)/(ii): a hand-made tree carrying a perfectly forged provenance block —
    the credential cycle 1 checked — is still refused, because the gate is the caller's
    intent, not a value in the file."""
    os.makedirs(os.path.join(scratch, "fold1"), exist_ok=True)
    wf._write_json(os.path.join(scratch, "fold_manifest.json"), {
        "task": "TASK_837", "fold_set": "real",
        "universe_db": "fixtures/trading_fixture.db",
        "_materializer": "TASK_837/walkforward.materialize",   # the cycle-1 sentinel
        wf._PROVENANCE_KEY: {"materializer": wf._MATERIALIZER_ID,
                             "universe_db": "fixtures/trading_fixture.db",
                             "db_kind": "fixture", "fold_set": "real",
                             "content_sha256": "0" * 64},
    })
    wf._write_json(os.path.join(scratch, "fold1", "period_train.json"), {"DECOY": True})
    before = wf._hash_tree(scratch)
    with pytest.raises(wf.MaterializeRefusal):
        _materialize_smoke(scratch)
    assert wf._hash_tree(scratch) == before


def test_b1_symlinked_manifest_refused_without_being_opened(scratch, tmp_path):
    """Escape (iii): a fold_manifest.json that is a SYMLINK is refused on file TYPE,
    before it is opened at all — one lstat/S_ISREG check, no taxonomy."""
    target = tmp_path / "elsewhere.json"
    target.write_text(json.dumps({"_materializer": "TASK_837/walkforward.materialize",
                                  wf._PROVENANCE_KEY: {"db_kind": "fixture"}}))
    os.symlink(str(target), os.path.join(scratch, "fold_manifest.json"))
    before = wf._hash_tree(scratch)
    with pytest.raises(wf.MaterializeRefusal) as ei:
        _materialize_smoke(scratch, allow_overwrite=True)   # even WITH the flag
    assert "not a regular file" in str(ei.value).lower() or "regular file" in str(ei.value)
    assert wf._hash_tree(scratch) == before


def test_fifo_manifest_refused_without_hanging(scratch):
    """MAJOR (TASK_891/897 hang class): a FIFO at fold_manifest.json must be refused on
    type, not opened. Without the lstat guard this call blocks forever."""
    os.mkfifo(os.path.join(scratch, "fold_manifest.json"))
    with pytest.raises(wf.MaterializeRefusal) as ei:
        _materialize_smoke(scratch, allow_overwrite=True)
    assert "regular file" in str(ei.value)


def test_unicode_garbage_manifest_gives_clean_refusal(scratch):
    """MINOR: a non-UTF-8 manifest raises UnicodeDecodeError (a ValueError, NOT a
    JSONDecodeError) — it must take the clean refusal path, not escape as a traceback."""
    with open(os.path.join(scratch, "fold_manifest.json"), "wb") as fh:
        fh.write(b"\xff\xfe\x00\x00 not utf-8 at all")
    with pytest.raises(wf.MaterializeRefusal) as ei:
        _materialize_smoke(scratch, allow_overwrite=True)
    assert "cannot be read as JSON" in str(ei.value)


def test_invalid_json_manifest_gives_clean_refusal(scratch):
    with open(os.path.join(scratch, "fold_manifest.json"), "w") as fh:
        fh.write("{ not json ")
    with pytest.raises(wf.MaterializeRefusal) as ei:
        _materialize_smoke(scratch, allow_overwrite=True)
    assert "cannot be read as JSON" in str(ei.value)


# ------------------------------------------- the provenance-driven REFUSALS ----

def test_fixture_over_real_refused_even_with_flag(scratch):
    """A tree whose provenance says REAL db must never be replaced with fixture-derived
    content, flag or no flag."""
    wf._write_json(os.path.join(scratch, "fold_manifest.json"), {
        "task": "TASK_837", "fold_set": "real", "universe_db": "trading_data.db",
        wf._PROVENANCE_KEY: {"materializer": wf._MATERIALIZER_ID,
                             "universe_db": "trading_data.db", "db_kind": "real",
                             "fold_set": "real", "content_sha256": "0" * 64},
    })
    before = wf._hash_tree(scratch)
    with pytest.raises(wf.MaterializeRefusal) as ei:
        _materialize_smoke(scratch, allow_overwrite=True)
    assert "REAL database" in str(ei.value)
    assert wf._hash_tree(scratch) == before


def test_inconsistent_provenance_refused_even_with_flag(scratch):
    """CYCLE 3 / F1 — the refusal no flag may override must not be overridable via the
    field it reads. A provenance block that names a REAL universe_db while claiming
    db_kind='fixture' (hand-edited or corrupt) previously took the fixture->fixture path
    and CLOBBERED real-provenance content, mislabelling itself in the diagnostic. The
    class is now always RE-DERIVED from universe_db and the disagreement is refused."""
    wf._write_json(os.path.join(scratch, "fold_manifest.json"), {
        "task": "TASK_837", "fold_set": "real", "universe_db": "trading_data.db",
        wf._PROVENANCE_KEY: {"materializer": wf._MATERIALIZER_ID,
                             "universe_db": "trading_data.db",   # REAL ...
                             "db_kind": "fixture",               # ... claims fixture
                             "fold_set": "real", "content_sha256": "0" * 64},
    })
    wf._write_json(os.path.join(scratch, "fold1", "period_train.json"), {"DECOY": True})
    before = wf._hash_tree(scratch)
    with pytest.raises(wf.MaterializeRefusal) as ei:
        _materialize_smoke(scratch, allow_overwrite=True)   # even WITH the flag
    msg = str(ei.value)
    assert "DISAGREES WITH ITSELF" in msg
    assert "'fixture'" in msg and "'real'" in msg, "the refusal must name BOTH classes"
    assert "trading_data.db" in msg
    assert wf._hash_tree(scratch) == before, "the refusal must leave the tree untouched"


def test_inconsistent_provenance_refused_in_the_other_direction(scratch):
    """The mirrored corruption (fixture universe_db claiming db_kind='real') is refused
    too — the check is 'the manifest disagrees with itself', not one favoured direction."""
    wf._write_json(os.path.join(scratch, "fold_manifest.json"), {
        "task": "TASK_837", "fold_set": "smoke",
        "universe_db": "fixtures/trading_fixture.db",
        wf._PROVENANCE_KEY: {"materializer": wf._MATERIALIZER_ID,
                             "universe_db": "fixtures/trading_fixture.db",
                             "db_kind": "real", "fold_set": "smoke",
                             "content_sha256": "0" * 64},
    })
    with pytest.raises(wf.MaterializeRefusal) as ei:
        _materialize_smoke(scratch, allow_overwrite=True)
    assert "DISAGREES WITH ITSELF" in str(ei.value)


def test_stored_db_kind_is_never_the_source_of_the_class(scratch):
    """Belt on F1's root cause: _prior_db_kind must IGNORE the stored string entirely
    when it agrees, and never return it."""
    prov = {"universe_db": "trading_data.db", "db_kind": "real"}
    assert wf._prior_db_kind(prov, "t")[1] == "real"
    assert wf._prior_db_kind({"universe_db": "fixtures/trading_fixture.db"}, "t")[1] \
        == "fixture"
    # No recorded universe_db == no provenance CLAIM: (None, None), so the class
    # comparison simply does not run. NOT "real" — inventing a class for a directory that
    # never recorded one would lock out every hand-made scratch out-root, and the tree
    # worth protecting is protected by the destination-identity fence, not by this.
    assert wf._prior_db_kind(None, "t") == (None, None)
    assert wf._prior_db_kind({}, "t") == (None, None)
    # ... but a db_kind claim with no universe_db to back it is a self-contradiction.
    with pytest.raises(wf.MaterializeRefusal):
        wf._prior_db_kind({"db_kind": "real"}, "t")


def test_legacy_manifest_grandfathered_for_fixture_over_real(scratch):
    """A PRE-TASK_904A manifest (no _provenance block) still gets the fixture-over-real
    protection, reconstructed from universe_db — and, crucially, is NOT locked out when
    the direction is the legitimate one (real over fixture: see B2 test below)."""
    wf._write_json(os.path.join(scratch, "fold_manifest.json"), {
        "task": "TASK_837", "fold_set": "real", "universe_db": "trading_data.db",
        "generated": "2026-07-08T22:23:01+00:00"})
    with pytest.raises(wf.MaterializeRefusal) as ei:
        _materialize_smoke(scratch, allow_overwrite=True)
    assert "REAL database" in str(ei.value)


# ------------------------------------------------------ B2: the SPEC.md path works ----

def test_b2_spec_path_not_locked_out(scratch, tmp_path):
    """The tracked folds/real/ manifest carries NO sentinel and no _provenance block.
    Under cycle 1 that refused forever. Here: a legacy fixture-provenance tree, upgraded
    from a NON-fixture db (the SPEC.md direction — a COPY, never trading_data.db in
    anger), proceeds with the flag."""
    real_db_copy = str(tmp_path / "stands_in_for_trading_data.db")
    _stand_in_real_db(real_db_copy)   # a db PROVEN not to be the project fixture
    _materialize_smoke(scratch)                    # fixture provenance, like the tracked tree
    # strip the _provenance block: exactly the tracked 2026-07-08 manifest shape
    mp = os.path.join(scratch, "fold_manifest.json")
    m = json.load(open(mp))
    m.pop(wf._PROVENANCE_KEY)
    wf._write_json(mp, m)
    out = wf.materialize(real_db_copy, wf.SMOKE_FOLDS, "smoke", scratch,
                         n_assets=wf.SMOKE_PIT_N_ASSETS, min_days=wf.PIT_MIN_DAYS,
                         jobs_db="trading_data.db", allow_overwrite=True)
    assert out[wf._PROVENANCE_KEY]["db_kind"] == "real"


# ------------------------------------------------------------- smoke by construction ----

def test_smoke_never_passes_the_flag():
    """Gate (i)/(3): a bare `smoke` cannot write tracked fold content BY CONSTRUCTION —
    run_smoke's source contains no allow_overwrite=True and never names folds/real as a
    materialize target."""
    import inspect
    src = inspect.getsource(wf.run_smoke)
    assert "allow_overwrite" not in src, \
        "run_smoke must never authorize an overwrite"
    assert "materialize(db, REAL_FOLDS, \"real\", tmp_real_root" in src, \
        "run_smoke must materialize the real rails into the SCRATCH root, not real_root"


# ------------------------------------------ F3: the FIFO guard on EVERY manifest reader ----

def test_validate_cli_does_not_hang_on_a_fifo_manifest(scratch):
    """CYCLE 3 / F3 — `validate` (which every `materialize` CLI call ends with, and which
    runs standalone) opened fold_manifest.json bare: a FIFO there hung it forever
    (reproduced: exit 124 under `timeout 8`). It must refuse promptly instead — with a
    clean parseable message, exiting 1 (not the hang, and not any other non-zero code)."""
    os.mkfifo(os.path.join(scratch, "fold_manifest.json"))
    proc = subprocess.run(
        [sys.executable, os.path.join(_HERE, "walkforward.py"),
         "validate", "--fold-root", scratch],
        cwd=_REPO, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 1
    assert "regular file" in proc.stderr


def test_validate_api_refuses_fifo_manifest(scratch):
    os.mkfifo(os.path.join(scratch, "fold_manifest.json"))
    with pytest.raises(wf.LeakError) as ei:
        wf.validate_folds(scratch, wf.FIXTURE_DB, verbose=False)
    assert "regular file" in str(ei.value)


def test_aggregate_refuses_fifo_manifest(scratch):
    os.mkfifo(os.path.join(scratch, "fold_manifest.json"))
    with pytest.raises(wf.LeakError) as ei:
        wf.aggregate(scratch, {}, os.path.join(scratch, "out.json"), tag="t")
    assert "regular file" in str(ei.value)


def test_guarded_reader_is_the_only_manifest_reader():
    """One implementation (the 870 remainder's rule): no manifest reader may re-open
    fold_manifest.json bare."""
    import inspect
    for fn in (wf.validate_folds, wf.aggregate, wf._read_manifest_provenance):
        src = inspect.getsource(fn)
        assert "_load_json_regular_file" in src, \
            f"{fn.__name__} does not use the shared guarded manifest reader"
        assert "open(man" not in src and "open(os.path.join(fold_root, \"fold_manifest" \
            not in src, f"{fn.__name__} still opens fold_manifest.json bare"


# --------------------------------------- TASK_904A MAJOR fix: aggregate on no manifest ----

def test_aggregate_refuses_missing_manifest_cleanly(tmp_path):
    """RUN-42 MAJOR: aggregate() on a missing fold_manifest.json used to raise a raw
    FileNotFoundError traceback — indistinguishable from a crash to an orchestrator that
    only knows how to parse a `LEAK GATE:` refusal. It must refuse the same way
    validate_folds does (os.path.lexists pre-check -> LeakError)."""
    empty_root = str(tmp_path / "no_manifest_here")
    os.makedirs(empty_root)
    with pytest.raises(wf.LeakError) as ei:
        wf.aggregate(empty_root, {}, os.path.join(empty_root, "out.json"), tag="t")
    assert "no fold_manifest.json" in str(ei.value)
    assert "under" in str(ei.value)


# ----------------------------- RUN-42 BLOCKER fix: the canonical-tree IDENTITY FENCE ----

def test_anchor_is_derived_from_the_destination_not_from_this_script(tmp_path):
    """**THE test whose absence hid BLOCKER 1.** No monkeypatching of any kind: the anchor
    is exercised for real, against a canonical real-fold tree belonging to a DIFFERENT
    checkout than the one this script lives in.

    Under the ``__file__``-relative anchor this call was completely unguarded — the constant
    named THIS repo's folds/real, the destination is another checkout's, the string
    comparison failed, and the fence went inert. Destination-first it fires, because the
    identity is resolved by walking up from the DESTINATION to its own ``.git``.

    The fake checkout's folds/real is EMPTY on purpose: an empty out-root needs no flag
    under every other rule in the module, so only the identity fence can refuse here."""
    _repo, canonical = _fake_repo(str(tmp_path), "other_checkout", populate=False)
    with pytest.raises(wf.MaterializeRefusal) as ei:
        _materialize_real(canonical, allow_overwrite=True)
    msg = str(ei.value)
    assert "canonical tracked real-fold tree" in msg
    assert "st_dev,st_ino" in msg, "the refusal must say how identity was established"
    assert not os.listdir(canonical), "the refusal must have written nothing"


def test_identity_fence_mutation_proof_both_directions(tmp_path, monkeypatch):
    """Mutation proof, on a throw-away /tmp COPY — never the tracked tree.

    Direction 1 (fence PRESENT, REAL derivation, no monkeypatch): the run-42 accident
    (fixture db + --fold-set real + --overwrite-existing, against a tree whose OWN manifest
    self-reports universe_db=fixture) is REFUSED and the copy is byte-for-byte unchanged.
    Direction 2 (fence REMOVED — ``_canonical_real_tree_hit`` stubbed to a no-op, which is
    what deleting the fence amounts to): the SAME call proceeds and rewrites the copy. That
    is the accident reproducing, and it is why the fence is load-bearing."""
    _repo, canonical = _fake_repo(str(tmp_path), "victim_checkout", populate=True)
    before = wf._hash_tree(canonical)

    with pytest.raises(wf.MaterializeRefusal) as ei:
        _materialize_real(canonical, allow_overwrite=True)
    assert "canonical tracked real-fold tree" in str(ei.value)
    assert wf._hash_tree(canonical) == before, "the refusal must leave the tree untouched"

    monkeypatch.setattr(wf, "_canonical_real_tree_hit", lambda dest: None)
    _materialize_real(canonical, allow_overwrite=True)          # must NOT raise
    assert wf._hash_tree(canonical) != before, (
        "MUTATION PROOF: with the destination-identity fence removed, nothing else in the "
        "module refuses fixture-derived content over this tree — the class comparison "
        "cannot, because the tree's OWN manifest self-reports universe_db=fixture. The "
        "clobber reproduces here on a /tmp copy, never against the tracked tree.")

    monkeypatch.undo()                                          # fence RESTORED
    with pytest.raises(wf.MaterializeRefusal):
        _materialize_real(canonical, allow_overwrite=True)


def test_blocker2_a_legal_non_real_fold_set_no_longer_disables_the_fence(tmp_path):
    """**BLOCKER 2.** ``--fold-set smoke`` is a perfectly legal argparse value. Under the
    caller-reported predicate that legality was the escape: ``smoke`` failed the fence's
    ``fold_set == "real"`` conjunct AND failed the plain fixture-over-real check (whose
    prior class is re-derived from the canonical tree's own manifest, which self-reports
    "fixture"), so 8-asset smoke-window content was written into the tracked 64-asset real
    rails. Destination-first, disagreement with the destination is DISQUALIFYING."""
    _repo, canonical = _fake_repo(str(tmp_path), "b2_checkout", populate=True)
    before = wf._hash_tree(canonical)
    with pytest.raises(wf.MaterializeRefusal) as ei:
        _materialize_real(canonical, fold_set="smoke", allow_overwrite=True)
    msg = str(ei.value)
    assert "'smoke'" in msg and "not 'real'" in msg, \
        "the refusal must name the disqualifying fold_set"
    assert wf._hash_tree(canonical) == before


def test_blocker2_isolated_only_the_fold_set_disagrees(tmp_path):
    """The same escape with EVERY other conjunct satisfied — a real-classed db and the
    explicit flag — so the only thing refusing is the fold-set disagreement itself. This is
    BLOCKER 2 with the db-class variable held constant."""
    _repo, canonical = _fake_repo(str(tmp_path), "b2_iso_checkout", populate=True)
    real_db = str(tmp_path / "stands_in_for_trading_data.db")
    _stand_in_real_db(real_db)
    before = wf._hash_tree(canonical)
    with pytest.raises(wf.MaterializeRefusal) as ei:
        _materialize_real(canonical, db=real_db, fold_set="smoke", allow_overwrite=True)
    assert "not 'real'" in str(ei.value)
    assert wf._hash_tree(canonical) == before


def test_full_consistent_unlock_is_not_locked_out(tmp_path):
    """The positive control the fence must not break: the SPEC.md-scheduled path — a
    real-classed --db, --fold-set real, and the explicit flag, all AGREEING with the
    destination — proceeds into the canonical tree. Fail-closed must not mean
    fail-always."""
    _repo, canonical = _fake_repo(str(tmp_path), "spec_path_checkout", populate=True)
    real_db = str(tmp_path / "stands_in_for_trading_data.db")
    _stand_in_real_db(real_db)
    out = _materialize_real(canonical, db=real_db, allow_overwrite=True)
    assert out[wf._PROVENANCE_KEY]["db_kind"] == "real"
    assert os.path.exists(os.path.join(canonical, "fold1", "universe_pit.json"))


def test_a_tmp_scratch_copy_outside_any_checkout_is_not_fenced(tmp_path):
    """The other direction of the same rule, and the reason destination-first is usable:
    a scratch copy of the tree that lives OUTSIDE any git checkout is NOT the canonical
    tracked tree and must remain freely writable. Over-fencing here would make every /tmp
    reconstruction and every scratch out-root unusable."""
    scratch_copy = str(tmp_path / "plain_scratch" / "folds" / "real")
    os.makedirs(os.path.dirname(scratch_copy), exist_ok=True)
    shutil.copytree(TRACKED_REAL, scratch_copy)
    assert wf._canonical_real_tree_hit(scratch_copy) is None
    before = wf._hash_tree(scratch_copy)
    _materialize_real(scratch_copy, allow_overwrite=True)        # must NOT raise
    assert wf._hash_tree(scratch_copy) != before


def test_symlinked_out_root_cannot_route_around_the_fence(tmp_path):
    """realpath first: pointing --out-root at a SYMLINK to the canonical tree is the same
    write, and is refused the same way. (This is the vector that clobbered 5 tracked files
    during run-42 verification.)"""
    _repo, canonical = _fake_repo(str(tmp_path), "symlink_checkout", populate=True)
    link = str(tmp_path / "innocent_looking_link")
    os.symlink(canonical, link)
    before = wf._hash_tree(canonical)
    with pytest.raises(wf.MaterializeRefusal) as ei:
        _materialize_real(link, allow_overwrite=True)
    assert "canonical tracked real-fold tree" in str(ei.value)
    assert wf._hash_tree(canonical) == before


def test_a_subdirectory_of_the_canonical_tree_is_fenced_too(tmp_path):
    """The fence covers the tree, not just its root entry: writing into
    ``folds/real/fold1`` would clobber tracked content just as surely."""
    _repo, canonical = _fake_repo(str(tmp_path), "subdir_checkout", populate=True)
    with pytest.raises(wf.MaterializeRefusal):
        _materialize_real(os.path.join(canonical, "fold1"), allow_overwrite=True)


def test_the_scratch_sibling_of_folds_real_is_not_mistaken_for_it(tmp_path):
    """A prefix bug here would break `smoke` outright: the per-run scratch root is
    ``folds/real_smoke_scratch_*`` — a STRING prefix of ``folds/real`` but a different
    directory. Containment must be path-component-wise, never ``startswith``."""
    _repo, canonical = _fake_repo(str(tmp_path), "sibling_checkout", populate=False)
    sibling = os.path.join(os.path.dirname(canonical), "real_smoke_scratch_abc")
    os.makedirs(sibling)
    assert wf._canonical_real_tree_hit(sibling) is None
    assert wf._canonical_real_tree_hit(os.path.join(sibling, "fold1", "x.json")) is None


def test_fence_fires_for_a_script_copy_run_from_a_foreign_location(tmp_path):
    """BLOCKER 1's exact reported vector, end to end at CLI level: the IDENTICAL unmodified
    ``walkforward.py``, copied to a foreign directory whose own repo-root walk finds
    nothing, invoked from a foreign cwd, pointed at another checkout's canonical tree by
    absolute path. Under the ``__file__``-relative anchor this was the inert case. The
    destination's folds/real is EMPTY, so nothing but the identity fence can refuse."""
    _repo, canonical = _fake_repo(str(tmp_path), "target_checkout", populate=False)
    copy_dir = os.path.join(str(tmp_path), "loose_copy", "experiments", "837_walkforward")
    os.makedirs(copy_dir)
    script = os.path.join(copy_dir, "walkforward.py")
    shutil.copyfile(os.path.join(_HERE, "walkforward.py"), script)
    env = dict(os.environ, PYTHONPATH=_REPO)   # `trader` importable; nothing else shared
    proc = subprocess.run(
        [sys.executable, script, "materialize", "--fold-set", "real",
         "--db", wf.FIXTURE_DB, "--out-root", canonical],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 1, (proc.stdout[-2000:], proc.stderr[-2000:])
    assert "MATERIALIZE REFUSED" in proc.stderr
    assert "canonical tracked real-fold tree" in proc.stderr
    assert not os.listdir(canonical), "the refusal must have written nothing"


def test_unresolvable_anchor_refuses_instead_of_going_inert(tmp_path, monkeypatch):
    """An inert fence that raises nothing is the worst outcome (that was BLOCKER 1's real
    severity). If identity cannot be established the guard must REFUSE."""
    def boom(_path):
        raise OSError("simulated: cannot resolve the destination's repository")
    monkeypatch.setattr(wf, "_repo_root_of", boom)
    with pytest.raises(wf.MaterializeRefusal) as ei:
        wf._canonical_real_tree_hit(str(tmp_path / "anywhere"))
    assert "cannot establish" in str(ei.value)


# --------------------------- the fence binds at the SHARED write choke point ----

def test_bare_corridor_into_canonical_tree_is_refused(tmp_path):
    """The 904B test ``test_bare_corridor_cli_leaves_tracked_real_tree_identical``,
    rewritten to run against a /tmp fake checkout. It FAILED against live code — bare
    ``corridor`` mutated the tracked tree, which is what moved the tracked tree hash twice
    in two days — and it PASSES now, because the identity fence binds at the shared write
    choke point rather than only inside ``materialize``. The corridor WRITER itself is
    unchanged (its two-phase redesign is TASK_904B)."""
    _repo, canonical = _fake_repo(str(tmp_path), "corridor_checkout", populate=True)
    before = wf._hash_tree(canonical)
    proc = subprocess.run(
        [sys.executable, os.path.join(_HERE, "walkforward.py"), "corridor",
         "--fold-set", "real", "--fold-root", canonical],
        cwd=_REPO, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 1, (proc.stdout[-2000:], proc.stderr[-2000:])
    assert "MATERIALIZE REFUSED" in proc.stderr
    assert "canonical tracked real-fold tree" in proc.stderr
    assert wf._hash_tree(canonical) == before, \
        "bare corridor must leave the canonical tree byte-identical"


def test_aggregate_into_canonical_tree_is_refused_by_the_choke_point(tmp_path):
    """``aggregate`` gets no unlock at all: its results belong in a run directory. The
    choke point refuses and says so. (Stub-vs-real summary detection remains 904B.)"""
    _repo, canonical = _fake_repo(str(tmp_path), "aggregate_checkout", populate=True)
    with pytest.raises(wf.MaterializeRefusal) as ei:
        wf._write_json(os.path.join(canonical, "walkforward_summary.json"), {"x": 1})
    assert "no authorized write scope is open" in str(ei.value)
    assert "runs/" in str(ei.value), "the refusal must point at the right destination"


def test_choke_point_unlock_is_scoped_and_closes_again(tmp_path):
    """The unlock covers ONE subtree and only for the duration of the writer's scope —
    it is not a global mode that a later, unauthorized write can ride on."""
    _repo, canonical = _fake_repo(str(tmp_path), "scope_checkout", populate=False)
    other = _fake_repo(str(tmp_path), "scope_other_checkout", populate=False)[1]
    real_db = str(tmp_path / "stands_in_for_trading_data.db")
    _stand_in_real_db(real_db)
    target = os.path.join(canonical, "probe.json")
    with wf._canonical_write_scope(canonical, real_db, "real", True, who="test"):
        wf._write_json(target, {"ok": True})               # inside the scope: allowed
        with pytest.raises(wf.MaterializeRefusal):         # a DIFFERENT canonical tree
            wf._write_json(os.path.join(other, "probe.json"), {"ok": True})
    assert wf._WRITE_UNLOCK == [], "the scope must not leak"
    with pytest.raises(wf.MaterializeRefusal):             # outside it again: refused
        wf._write_json(target, {"ok": True})


# ========================================================================================
# RUN-42 CYCLE 3 BLOCKER — the SECOND __file__ anchor, and the polarity collision
# ========================================================================================
# The unlock's "real-classed --db" conjunct asked a classifier whose fixture identity was
# ``FIXTURE_DB`` — ``__file__``-relative — and whose fallback for "cannot tell" was
# ``"real"``. Run from a worktree or a /tmp reconstruction (where the GITIGNORED fixture
# does not exist), an ABSOLUTE path to the actual project fixture matched neither
# ``samefile`` nor the ``_rel`` spelling, fell through to ``"real"``, completed the unlock,
# and overwrote a real-labelled canonical tree at exit 0 — the guard's own diagnostic
# printing "(real -> real)" about the project fixture. NONE of the 54 cycle-2 tests pointed
# ``--db`` by absolute path at the literal project fixture from a script COPY: that test gap
# was exactly the implementation gap, so it is closed first, at CLI level, in both writers.

def _foreign_checkout(base, name, with_pin=True, populate=True):
    """A fake checkout that reproduces the reported rig: a ``.git``, the canonical real-fold
    path, a ``fixtures/`` directory that (like every real checkout) does NOT contain the
    GITIGNORED ``trading_fixture.db``, and — when ``with_pin`` — the TRACKED content pin
    ``fixtures/FIXTURE_IDENTITY.json`` that a real checkout DOES carry. Also holds a COPY of
    ``walkforward.py`` under its own ``experiments/837_walkforward/``, so a subprocess run
    from it derives ``_REPO`` = this fake root: the exact "which copy of the script runs"
    condition the cycle-2 note claimed did not matter.

    Returns ``(root, canonical_real_dir, script_path)``."""
    root, canonical = _fake_repo(base, name, populate=populate)
    os.makedirs(os.path.join(root, "fixtures"), exist_ok=True)
    if with_pin:
        shutil.copyfile(os.path.join(_REPO, "fixtures", "FIXTURE_IDENTITY.json"),
                        os.path.join(root, "fixtures", "FIXTURE_IDENTITY.json"))
    script = os.path.join(root, "experiments", "837_walkforward", "walkforward.py")
    shutil.copyfile(os.path.join(_HERE, "walkforward.py"), script)
    return root, canonical, script


@pytest.mark.parametrize("cmd", ["materialize", "corridor"])
def test_script_copy_pointing_db_at_the_literal_project_fixture_is_refused(tmp_path, cmd):
    """THE missing test. A script COPY, run from a foreign checkout whose own
    ``fixtures/trading_fixture.db`` is absent, with ``--db`` an ABSOLUTE path to the LITERAL
    project fixture, ``--fold-set real`` and ``--overwrite-existing`` — i.e. every other
    conjunct satisfied, so only the db class can refuse. Under the ``__file__``-anchored
    classifier this exited 0 and clobbered the tree; here it must exit 1 with
    ``MATERIALIZE REFUSED`` and leave the destination byte-identical.

    Parametrized over BOTH writers: ``corridor`` reaches the same unlock through
    ``_canonical_write_scope`` and was clobbering the tracked tree by the same route."""
    _root, canonical, script = _foreign_checkout(str(tmp_path), f"foreign_{cmd}")
    before = wf._hash_tree(canonical)
    root_arg = (["--out-root", canonical] if cmd == "materialize"
                else ["--fold-root", canonical])
    proc = subprocess.run(
        [sys.executable, script, cmd, "--fold-set", "real",
         "--db", os.path.join(_REPO, "fixtures", "trading_fixture.db"),
         "--overwrite-existing"] + root_arg,
        cwd=str(tmp_path), env=dict(os.environ, PYTHONPATH=_REPO),
        capture_output=True, text=True, timeout=600)
    assert proc.returncode == 1, (proc.stdout[-3000:], proc.stderr[-3000:])
    assert "MATERIALIZE REFUSED" in proc.stderr
    assert "POSITIVELY PROVEN real-classed --db" in proc.stderr
    assert "NAMES the project fixture" in proc.stderr
    assert wf._hash_tree(canonical) == before, "the refusal must have written nothing"


def test_renamed_byte_identical_fixture_copy_is_not_positively_real(tmp_path):
    """Fixture identity by CONTENT, via the TRACKED pin — the source of truth that survives
    into a checkout where the gitignored fixture file does not. A byte-identical copy of the
    fixture under a real-sounding name defeats every path/inode comparison, so without the
    pin it would classify "real" and author the tree. It must be refused, and the refusal
    must say WHY (byte-identical to the pinned fixture)."""
    _root, canonical, _script = _foreign_checkout(str(tmp_path), "content_id_checkout")
    renamed = str(tmp_path / "definitely_the_real_trading_data.db")
    shutil.copyfile(wf.FIXTURE_DB, renamed)          # deliberately byte-identical
    before = wf._hash_tree(canonical)
    assert wf._db_class(renamed, canonical)[0] == wf._DB_FIXTURE
    with pytest.raises(wf.MaterializeRefusal) as ei:
        _materialize_real(canonical, db=renamed, allow_overwrite=True)
    assert "BYTE-IDENTICAL to the project fixture pinned in" in str(ei.value)
    assert wf._hash_tree(canonical) == before


def test_unestablishable_fixture_identity_refuses_to_authorize(tmp_path, monkeypatch):
    """Requirement 2's tail: if fixture identity cannot be established AT ALL, that is "not
    positively real", so REFUSE. Here neither candidate root offers evidence — the
    destination's checkout has no fixture file and no pin, and this script's ``_REPO`` is
    redirected to an empty directory — yet the ``--db`` is a genuinely non-fixture database.
    A classifier that answered "real" on no evidence is what cycle 3 was.

    This is the suite's ONE legitimate use of ``monkeypatch.setattr`` on an anchor besides
    the ``_repo_root_of`` mutation proof: ``_REPO`` is a process-level constant and the
    second candidate root cannot otherwise be emptied from inside the same process."""
    _root, canonical, _script = _foreign_checkout(str(tmp_path), "no_evidence_checkout",
                                                 with_pin=False)
    real_db = _stand_in_real_db(str(tmp_path / "an_actual_real_db.db"))
    empty_repo = str(tmp_path / "empty_repo")
    os.makedirs(empty_repo)
    monkeypatch.setattr(wf, "_REPO", empty_repo)
    before = wf._hash_tree(canonical)
    cls, why = wf._db_class(real_db, canonical)
    assert cls == wf._DB_UNPROVEN, (cls, why)
    assert wf._require_positively_real_db(real_db, canonical, "t") is not None
    with pytest.raises(wf.MaterializeRefusal) as ei:
        _materialize_real(canonical, db=real_db, allow_overwrite=True)
    assert "CANNOT BE ESTABLISHED" in str(ei.value)
    assert wf._hash_tree(canonical) == before


def test_the_two_db_polarities_are_separate_functions(tmp_path):
    """The STRUCTURAL half of the fix, asserted directly: the two consumers of the db class
    have OPPOSITE polarity, so they are separate, differently-named functions over an
    explicit tri-state — not one boolean-ish classifier a future reader can point the wrong
    way. On the SAME unprovable input the protective one answers "real" (protect harder) and
    the authorizing one REFUSES (unknown is not a permit)."""
    unprovable = "/nowhere/at/all/definitely_not_here.db"
    assert wf._db_kind_protective(unprovable) == wf._DB_REAL       # protect harder
    assert wf._require_positively_real_db(unprovable, str(tmp_path), "t") is not None
    # ... and the tri-state itself distinguishes the three cases the two-valued classifier
    # could not, which is what makes the two collapses expressible at all.
    assert wf._db_class(wf.FIXTURE_DB)[0] == wf._DB_FIXTURE
    assert wf._db_class(unprovable)[0] == wf._DB_UNPROVEN
    assert wf._db_class(os.path.join(_REPO, "fixtures", "make_fixture.py"))[0] == wf._DB_REAL
    assert len({wf._DB_FIXTURE, wf._DB_REAL, wf._DB_UNPROVEN}) == 3


# ----------------------------------------- MAJOR: a falsy --db must not go inert ----

@pytest.mark.parametrize("db", ["", None])
def test_falsy_db_refused_as_a_materialize_refusal(scratch, db):
    """``_db_kind`` returned a THIRD value ("unknown") for a falsy path while both refusal
    branches gated on ``== "fixture"``, so ``--db ""`` disabled them. It failed closed only
    by accident (``_ro_connect("")`` raising sqlite3.OperationalError), and the ``None``
    variant died on a TypeError inside the guard's OWN diagnostic print — neither carrying
    the parseable prefix an orchestrator reads."""
    with pytest.raises(wf.MaterializeRefusal) as ei:
        _materialize_real(scratch, db=db, allow_overwrite=True)
    assert "REFUSING" in str(ei.value)


def test_falsy_db_refused_against_the_canonical_tree_too(tmp_path):
    _repo, canonical = _fake_repo(str(tmp_path), "falsydb_checkout", populate=True)
    before = wf._hash_tree(canonical)
    with pytest.raises(wf.MaterializeRefusal):
        _materialize_real(canonical, db="", allow_overwrite=True)
    assert wf._hash_tree(canonical) == before


def test_nonexistent_db_refused_rather_than_classified(scratch, tmp_path):
    """A db path that does not exist cannot be classified, so the refusals that depend on
    the class would be inert. Refuse instead."""
    with pytest.raises(wf.MaterializeRefusal) as ei:
        _materialize_real(scratch, db=str(tmp_path / "not_a_db.sqlite"),
                          allow_overwrite=True)
    assert "not an existing regular file" in str(ei.value)


def test_db_kind_protective_has_exactly_two_values_and_matches_its_docstring():
    """Code/docstring agreement for the PROTECTIVE polarity: exactly two values, and
    "cannot classify" collapses to "real" so a directory gets protected HARDER. Renamed
    from ``_db_kind`` in run-42 cycle 3 — the bare name served both polarities and the
    rename is half the fix (see ``test_the_two_db_polarities_are_separate_functions``)."""
    assert not hasattr(wf, "_db_kind"), \
        "the polarity-ambiguous _db_kind must not survive under its old name"
    assert wf._db_kind_protective(None) == "real"
    assert wf._db_kind_protective("") == "real"
    assert wf._db_kind_protective(wf.FIXTURE_DB) == "fixture"
    assert wf._db_kind_protective("trading_data.db") == "real"
    assert wf._db_kind_protective("/nowhere/at/all.db") == "real"


# ------------------- MAJOR: the shared reader must type-check what it returns ----

_WRONG_TYPE_MANIFESTS = {"list": "[1, 2, 3]", "null": "null", "string": '"hello"'}


@pytest.mark.parametrize("shape", sorted(_WRONG_TYPE_MANIFESTS))
def test_wrong_type_manifest_refused_by_validate(scratch, shape):
    """Valid JSON of the wrong TYPE used to escape the guarded reader and blow up
    downstream as ``AttributeError: 'list' object has no attribute 'get'`` — a raw
    traceback with no ``LEAK GATE:`` prefix, which falsified the claim that every
    guarded-reader refusal path exits 1 through the handlers."""
    with open(os.path.join(scratch, "fold_manifest.json"), "w") as fh:
        fh.write(_WRONG_TYPE_MANIFESTS[shape])
    with pytest.raises(wf.LeakError) as ei:
        wf.validate_folds(scratch, wf.FIXTURE_DB, verbose=False)
    assert "WRONG TYPE" in str(ei.value)


@pytest.mark.parametrize("shape", sorted(_WRONG_TYPE_MANIFESTS))
def test_wrong_type_manifest_refused_by_aggregate(scratch, shape):
    with open(os.path.join(scratch, "fold_manifest.json"), "w") as fh:
        fh.write(_WRONG_TYPE_MANIFESTS[shape])
    with pytest.raises(wf.LeakError) as ei:
        wf.aggregate(scratch, {}, os.path.join(scratch, "out.json"), tag="t")
    assert "WRONG TYPE" in str(ei.value)


@pytest.mark.parametrize("cmd", ["validate", "aggregate"])
def test_wrong_type_manifest_exits_1_with_a_parseable_prefix(scratch, cmd):
    """At CLI level, which is where the falsified claim lived: exit 1 and a ``LEAK GATE:``
    prefix, not a traceback."""
    with open(os.path.join(scratch, "fold_manifest.json"), "w") as fh:
        fh.write("[1, 2, 3]")
    proc = subprocess.run(
        [sys.executable, os.path.join(_HERE, "walkforward.py"), cmd,
         "--fold-root", scratch],
        cwd=_REPO, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 1, (proc.stdout[-2000:], proc.stderr[-2000:])
    assert "LEAK GATE:" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_wrong_type_provenance_manifest_refused_by_materialize(scratch):
    with open(os.path.join(scratch, "fold_manifest.json"), "w") as fh:
        fh.write("[1, 2, 3]")
    with pytest.raises(wf.MaterializeRefusal) as ei:
        _materialize_smoke(scratch, allow_overwrite=True)
    assert "WRONG TYPE" in str(ei.value)


# ------------------------------------------------- stale scratch-dir sweep (crash debris) ----

def _mk_scratch_debris(folds_dir, name, age_secs):
    """Debris aged CONSISTENTLY — the whole tree, not just the top-level dir. Since the
    sweep ages on the newest mtime anywhere inside (cycle 3 / F4), a tree with fresh
    nested files is a LIVE run by definition, whatever its top-level dir says."""
    d = os.path.join(folds_dir, name)
    os.makedirs(os.path.join(d, "fold1"), exist_ok=True)
    with open(os.path.join(d, "fold1", "period_train.json"), "w") as fh:
        fh.write("{}")
    t = time.time() - age_secs
    for p in (os.path.join(d, "fold1", "period_train.json"),
              os.path.join(d, "fold1"), d):
        os.utime(p, (t, t))
    return d


def test_sweep_removes_stale_debris_but_spares_a_concurrent_run(tmp_path):
    folds = str(tmp_path / "folds")
    os.makedirs(folds)
    stale = _mk_scratch_debris(folds, "real_smoke_scratch_KILLED", 7 * 3600)
    fresh = _mk_scratch_debris(folds, "real_smoke_scratch_INFLIGHT", 30)
    keep_real = os.path.join(folds, "real")
    keep_smoke = os.path.join(folds, "smoke")
    os.makedirs(keep_real)
    os.makedirs(keep_smoke)
    removed = wf._sweep_stale_scratch(folds)
    assert not os.path.exists(stale), "a crashed run's debris must be reaped"
    assert os.path.isdir(fresh), "a CONCURRENT run's scratch dir must be spared"
    assert os.path.isdir(keep_real) and os.path.isdir(keep_smoke), \
        "the sweep must never touch folds/real or folds/smoke"
    assert len(removed) == 1


def test_sweep_never_follows_a_symlink_out_of_the_pattern(tmp_path):
    """The sweep must not be able to reach anything outside the gitignored pattern."""
    folds = str(tmp_path / "folds")
    outside = tmp_path / "precious"
    outside.mkdir()
    (outside / "keep.txt").write_text("do not delete me")
    os.makedirs(folds)
    link = os.path.join(folds, "real_smoke_scratch_EVIL")
    os.symlink(str(outside), link)
    t = time.time() - 99 * 3600
    os.utime(link, (t, t), follow_symlinks=False)
    assert wf._sweep_stale_scratch(folds) == []
    assert (outside / "keep.txt").exists(), "the sweep followed a symlink — unsafe"
    assert os.path.islink(link), "the symlink itself is not the sweep's business"


def test_sweep_spares_a_dir_whose_only_recent_writes_are_NESTED(tmp_path):
    """CYCLE 3 / F4 (MAJOR) — the tail of a real-fold smoke run writes only into
    ``<scratch>/fold*/`` and then only READS. Writing into a subdirectory does NOT bump
    the parent's mtime, so ageing on the top-level dir alone let a sibling run rmtree a
    LIVE peer's working directory. Age on the newest mtime anywhere in the tree."""
    folds = str(tmp_path / "folds")
    os.makedirs(folds)
    d = os.path.join(folds, "real_smoke_scratch_INFLIGHT_TAIL")
    os.makedirs(os.path.join(d, "fold1"))
    old = time.time() - 9 * 3600
    os.utime(d, (old, old))                     # top-level dir looks 9h stale ...
    inner = os.path.join(d, "fold1", "corridor.json")
    with open(inner, "w") as fh:                # ... but the run is writing RIGHT NOW
        fh.write("{}")
    assert time.time() - os.lstat(d).st_mtime > wf._SCRATCH_STALE_SECS, \
        "precondition: the top-level mtime must look stale"
    assert wf._sweep_stale_scratch(folds) == [], "a live peer was reaped"
    assert os.path.exists(inner)


def test_sweep_refuses_a_symlinked_folds_root(tmp_path):
    """CYCLE 3 / F5 — per-entry lstat refuses a symlinked CHILD, but os.path.isdir /
    os.listdir follow a symlinked ROOT, so an aged same-named subdir in the target got
    rmtree'd. lstat the root too."""
    real_folds = tmp_path / "elsewhere"
    real_folds.mkdir()
    victim = real_folds / "real_smoke_scratch_VICTIM"
    victim.mkdir()
    (victim / "keep.txt").write_text("do not delete me")
    t = time.time() - 99 * 3600
    os.utime(str(victim / "keep.txt"), (t, t))
    os.utime(str(victim), (t, t))
    link = str(tmp_path / "folds")
    os.symlink(str(real_folds), link)
    assert wf._sweep_stale_scratch(link) == []
    assert (victim / "keep.txt").exists(), "the sweep followed a symlinked folds root"


def test_debris_does_not_refuse_or_promote(tmp_path, scratch):
    """The failure modes the flow-verifier proved closed must STAY closed with debris
    present: debris must not cause a refusal, must not be silently promoted into the
    tracked tree, and must not be read as this run's output."""
    folds = os.path.join(_HERE, "folds")
    debris = _mk_scratch_debris(folds, "real_smoke_scratch_DEBRIS_t904a", 30)
    try:
        before = wf._hash_tree(TRACKED_REAL)
        m = _materialize_smoke(scratch)          # a DIFFERENT, fresh scratch dir
        assert m["fold_set"] == "smoke"          # no refusal
        assert wf._hash_tree(TRACKED_REAL) == before   # nothing promoted
        assert os.path.isdir(debris)             # debris untouched by this run
    finally:
        shutil.rmtree(debris, ignore_errors=True)


def test_default_materialize_cli_refuses_over_tracked_real_tree():
    """The accident the task names: the bare CLI default
    (`materialize` -> --db fixture --fold-set real --out-root folds/real) must refuse,
    exit non-zero, and leave the tracked tree byte-identical."""
    before = wf._hash_tree(TRACKED_REAL)
    proc = subprocess.run(
        [sys.executable, os.path.join(_HERE, "walkforward.py"), "materialize"],
        cwd=_REPO, capture_output=True, text=True, timeout=300)
    assert proc.returncode != 0, proc.stdout[-2000:]
    assert "MATERIALIZE REFUSED" in proc.stderr
    assert "--overwrite-existing" in proc.stderr
    assert wf._hash_tree(TRACKED_REAL) == before, \
        "the tracked folds/real/ tree must be byte-identical after a refused materialize"
