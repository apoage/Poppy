"""TASK_888 — the ONE checkpoint resolution function, and a loader that refuses.

D38's ruling on why this module exists: two defects reported separately turned out to be the
same defect. (1) **Path identity** — a `realpath` guard checking whether a checkpoint is
loadable, next to an `abspath` loader actually reading it, can disagree about which directory a
`--checkpoint` argv string names. This repo SHIPS exactly the symlinks that expose it
(`jax_ckpt_fast`, `jax_ckpt_easy` -> `/mnt/scratch/...`): a path that crosses one of
them and then `..` resolves to a DIFFERENT directory under `abspath` (lexical `..` cancellation,
symlink not resolved) than under `realpath` (the symlink is resolved FIRST, then `..` applies to
its real parent) — the guard can bless directory A while the loader opens directory B. (2) **The
production fail-open** — flax's own `restore_checkpoint` returns its `target` argument BACK,
UNCHANGED, when it finds nothing loadable at a path (measured on flax 0.10.7): not a raise, not
a `None`. Every caller that checked `if loaded is not None` (or nothing at all) before printing
"loaded successfully" was fail-OPEN: a nonexistent/unloadable `--checkpoint` silently continues
on the fresh-init weights the trainer already had, under a message that says otherwise.

Both are fixed by the same thing: resolve a `--checkpoint` string to a real, loadable directory
with ONE function (`resolve_checkpoint_dirs`, realpath-based), and make the loader
(`load_checkpoint_into`) REFUSE — never continue — when nothing loadable is there, using that
SAME resolved directory for the actual `trainer.load_checkpoint()` call so the path the guard
verified is byte-for-byte the path the loader opens.

Ported from the TASK_886 stash (`experiments/835_stage4_eval/eval_vector.py`, cycle 2/3 —
`_resolve_checkpoint_dirs` et al.), which is where this evidence was originally gathered
(three strace-proven infinite hangs on a FIFO planted inside a checkpoint leaf; the numeric
param-equality detector that false-positived on an honestly-saved fresh-init checkpoint and
broke `861/smoke_e2e` + `835/scorecard_smoke`). Generalized here into a module every checkpoint
LOAD site can import (`eval_vector.py`, `train_jax_fast.py`, `experiments/857_vhat_distill/
distill.py`, `real_trade_jax.py`), instead of four sites each holding their own copy that can
drift from the others exactly the way the guard drifted from the loader. Paths returned here are
ABSOLUTE (realpath); callers relativize for display if they want a repo-relative string.

NOT ported (deliberately): TASK_886's certification CONTRACT (`_certification_verdict`,
`_assert_provenance_binds_loaded_params`, the universe-pin machinery). That ships separately;
welding it to this load-path hardening is exactly what halted TASK_886 (D38).
"""
import hashlib
import os
import stat

import numpy as np
import jax
from flax.training import checkpoints as flax_checkpoints


# =============================================================================
# refusal hygiene — an unbounded echo is its own defect
# =============================================================================

# A refusal message is read by a human and pasted into a LEDGER row; nothing downstream parses
# it. The operator-supplied `--checkpoint` string is bounded only by MAX_ARG_STRLEN (~128KB per
# argv entry on Linux), and every refusal below interpolates it at least once, sometimes beside
# an equally long resolved path — measured before this cap: a ~130,000-char argv produced a
# 260,505-char exception and a 390,559-char log line, and an 800,090-char one produced
# 1,600,687 chars. Clipping the ECHO does not weaken the refusal (the refusal already happened;
# this is only how it is spelled) and it keeps a fail-closed path from turning into a log flood.
_REFUSAL_ECHO_MAX = 300


def clip_for_refusal(value, limit=_REFUSAL_ECHO_MAX, quote=True):
    """`repr(value)` (or `str(value)` with `quote=False`), clipped to `limit` characters with an
    explicit, counted marker.

    Used for EVERY operator-supplied or filesystem-derived string interpolated into a refusal.
    The marker names how much was dropped, so a clipped echo can never be mistaken for the whole
    string (a silently truncated path is worse than an obviously truncated one). `quote=False`
    is for re-embedding an already-formatted multi-line message (an exception body), where
    `repr` would escape every newline into one unreadable line."""
    text = repr(value) if quote else str(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [clipped, {len(text)} chars total]"


# =============================================================================
# path identity — the ONE resolution function (scope 1)
# =============================================================================

def flax_is_orbax_checkpoint(path):
    """`flax.training.checkpoints._is_orbax_checkpoint` — behind a NAMED failure.

    It is a PRIVATE (underscore) flax symbol. `requirements.txt` asks only for `flax>=0.7.0`
    (only `requirements.frozen.txt` pins 0.10.7), so a legitimate install can lack it. Without
    this guard the program dies several frames inside a resolver with a bare `AttributeError`;
    raise the version contract by name instead."""
    fn = getattr(flax_checkpoints, "_is_orbax_checkpoint", None)
    if fn is None:
        raise RuntimeError(
            f"flax.training.checkpoints._is_orbax_checkpoint is MISSING on the installed flax "
            f"({getattr(__import__('flax'), '__version__', 'unknown')}).\n"
            f"  checkpoint_io resolves --checkpoint with the SAME function flax's own "
            f"restore_checkpoint uses to decide what to load, rather than re-deriving 'is this "
            f"an orbax checkpoint' from filename patterns (a second copy drifts from the loader "
            f"it guards — TASK_888's whole thesis, one layer down). It is a PRIVATE flax "
            f"symbol, so an upgrade/downgrade can remove it.\n"
            f"  FIX: install the pinned flax (requirements.frozen.txt: flax==0.10.7).")
    return fn(path)


# The marker files flax's `_is_orbax_checkpoint` probes for — read off flax where available so
# the two lists cannot drift, literal names as the fallback. This is what gets STAT'ed before
# anything OPENS them.
_ORBAX_MARKERS = tuple(
    getattr(flax_checkpoints, name, default) for name, default in
    (("ORBAX_CKPT_FILENAME", "checkpoint"),
     ("ORBAX_METADATA_FILENAME", "_METADATA"),
     ("ORBAX_MANIFEST_OCDBT", "manifest.ocdbt")))


def resolve_checkpoint_dirs(path):
    """Resolve a `--checkpoint` argv string to `(run_dir, leaf_dir)` using flax's OWN checkpoint
    detection (`_is_orbax_checkpoint` / `latest_checkpoint`) — the EXACT functions
    `trainer.load_checkpoint`'s `restore_checkpoint` uses internally to decide what to load.
    THE hard constraint this module exists to satisfy: judge checkpoint identity by the same
    function the loader uses, never a re-implementation that can drift from it.

    ONE DISCLOSED PLACE WHERE THAT REUSE IS NARROWER THAN FLAX (fix-pass M5 — stated because an
    overstated claim is itself a defect in this module). The functions called here are flax's,
    but flax's `restore_checkpoint` ALSO has an unconditional non-directory branch: handed a path
    that is a regular FILE, it loads it as a legacy single-file msgpack checkpoint. This
    function does not reach that branch — `_is_orbax_checkpoint` is False for a regular file, and
    `latest_checkpoint` then raises `NotADirectoryError` (an `OSError`, caught below), so
    `leaf_dir` comes back `None` and every caller REFUSES. Naming the run DIR that holds such a
    leaf resolves correctly; only naming the leaf FILE itself refuses. Measured, and left as-is
    deliberately: it is unreachable in this repo (all 117 `checkpoint_*` leaves on disk are orbax
    DIRECTORIES — this program has never written a legacy single-file checkpoint), and it errs
    fail-CLOSED, which is the safe direction. Adding a legacy-load code path would be new,
    untested load surface for an input that does not exist here.

    Two committed `--checkpoint` spellings name the SAME physical checkpoint: the run dir passed
    to `trainer.save_checkpoint` (e.g. `.../ckpt_foo`) — flax writes the real orbax data one
    level deeper, at `.../ckpt_foo/checkpoint_<step>` — or that orbax leaf directly. Both resolve
    to `run_dir=".../ckpt_foo"` here (scope 1 / gate i); provenance and `trainer_metadata.json`
    siblings live beside the RUN dir, not beside the leaf.

    `realpath`, not `abspath` (the D38 divergence): this repo SYMLINKS checkpoint storage
    (`jax_ckpt_fast`/`jax_ckpt_easy` -> /mnt/scratch). Under `abspath`, a path crossing
    one of those symlinks and then `..` cancels lexically back to the symlink's OWN parent
    directory (the symlink target is never consulted); under `realpath` it resolves the symlink
    FIRST and then applies `..` to the REAL parent — a different directory. Whichever function
    reads the checkpoint must use the SAME resolution as whichever function decides it is safe
    to read, or a guard verdict about checkpoint A ships stamped on a run that loaded checkpoint
    B. This function IS that one resolution, called by both `preflight_checkpoint_loadable`
    (the guard) and `load_checkpoint_into` (the loader) — never re-derived at either call site.

    `leaf_dir` is the concrete orbax/flax checkpoint directory `restore_checkpoint` would use for
    a `step=None` load, or `None` if none exists there (the pre-load "is this loadable" probe). A
    pure filesystem read: nothing is loaded, and nothing is OPENED (see `unsafe_tree_defect` for
    why that distinction matters).

    CORRECTOR: resolves, never repairs — an unresolvable path returns `run_dir=realpath(path),
    leaf_dir=None`; the caller decides what to do with that."""
    realpath = os.path.realpath(path)
    try:
        is_leaf = flax_is_orbax_checkpoint(realpath)
    except (OSError, ValueError):
        is_leaf = False
    if is_leaf:
        return (os.path.dirname(realpath) or realpath), realpath
    try:
        leaf = flax_checkpoints.latest_checkpoint(realpath, prefix="checkpoint_")
    except (OSError, ValueError):
        leaf = None
    return realpath, leaf


# =============================================================================
# hang-safety — stat before open, never the reverse (886 cycle 2/3 evidence)
# =============================================================================

def unsafe_to_open_defect(path, max_bytes, what):
    """`None` if `path` is a regular file small enough to slurp, else a human-readable defect
    phrase. The caller establishes PRESENCE first (`os.path.lexists`) — absence is the caller's
    own state, never this helper's.

    ONE definition of "safe to open", for every sidecar this program reads off a path it DERIVED
    from an operator-supplied string. `stat` BEFORE `open`: a FIFO/socket/device blocks the
    reader FOREVER (measured: a FIFO hung a grading run 45s+ until killed), a directory is
    present-but-unusable, and an unbounded file is an unbounded slurp under the RAM cap. A hang
    is not a verdict.

    READABILITY IS PART OF "SAFE TO OPEN" (fix-pass M4). Type and size say the open will not
    block and will not exhaust RAM; they do not say it will SUCCEED. A `chmod 000` file passes
    `S_ISREG` and the size cap and then fails inside whatever library opens it — for a checkpoint
    payload that surfaces from orbax/tensorstore as a bare `ValueError`, NOT a
    `CheckpointRefusalError`, so callers narrowly catching this module's own refusal (e.g.
    `tools/paper_trade_vectorbt.py`) do not catch it and the refusal stops being uniform.
    `os.access(R_OK)` is checked here so that class refuses by the same named path as every
    other defect. It is a best-effort probe with the same disclosed TOCTOU remainder as the
    stat above (and, like any permission check, it answers for the CURRENT euid — as root it
    is vacuously true)."""
    try:
        st = os.stat(path)
    except OSError as e:
        return (f"it is present but cannot be stat'ed "
                f"({type(e).__name__}: {clip_for_refusal(e, quote=False)})")
    if not stat.S_ISREG(st.st_mode):
        return ("it is present but NOT a regular file (FIFO/socket/device/directory) — "
                "refusing to open it: a FIFO or device here blocks the reader forever, "
                "which is a hang, not a verdict")
    if st.st_size > max_bytes:
        return (f"it is {st.st_size} bytes — {what} is ~KB; refusing to "
                f"read anything over {max_bytes} bytes into memory")
    if not os.access(path, os.R_OK):
        return ("it is a regular file but is NOT READABLE by this process — refusing to hand "
                "it to the loader: the open would fail several frames inside the reading "
                "library as an untyped error instead of a named refusal here")
    return None


# The two sidecars `trainer.load_checkpoint` OPENS, which live beside the RUN dir — one level
# ABOVE the orbax leaf, so the leaf walk below can never see them. `trainer_metadata.json` is
# 281 bytes and `popart_state.npz` 1276 bytes for every real checkpoint in this repo, so the cap
# carries ~3 orders of magnitude of headroom; it exists only so a hostile/corrupt sidecar cannot
# become an unbounded slurp under the RAM cap.
RUN_DIR_SIDECARS = ("trainer_metadata.json", "popart_state.npz")
SIDECAR_MAX_BYTES = 4 * 1024 * 1024


def leaf_metadata_path_for(run_dir, step):
    """``<run_dir>/trainer_metadata_<step>.json`` — the PER-LEAF architecture/fingerprint record
    written beside each ``checkpoint_<step>`` leaf (TASK_889 fix pass).

    WHY IT EXISTS. ``trainer_metadata.json`` is ONE file per RUN DIR, overwritten by the most
    recent ``save_checkpoint``. A run dir normally holds SEVERAL leaves (``keep=3`` is the
    default and what ``train_jax_fast.py`` uses), so for every leaf that is not the newest, the
    shared sidecar describes a DIFFERENT save. `param_fingerprint` is genuinely leaf-specific,
    so it cannot be validated for those leaves from the shared sidecar at all — and a loader
    that responds to that by SKIPPING the check turns "cannot verify" into "proceed", which is
    the D37/D39 pathology (measured: an older leaf's architecture mismatch and a tampered older
    leaf's payload both escaped a step-gated skip, through the production API). One record per
    leaf removes the ambiguity instead of tolerating it.

    WHY THIS NAME, NOT ``checkpoint_<step>.trainer_metadata.json`` (MEASURED, flax 0.10.7).
    flax's `latest_checkpoint` globs ``<run_dir>/checkpoint_*`` and takes the natural-sort LAST
    entry — it never checks that what it found is a directory. A sibling named
    ``checkpoint_100.trainer_metadata.json`` is therefore RETURNED AS THE CHECKPOINT:

        latest_checkpoint(run_dir)                     -> .../checkpoint_100
        (plant checkpoint_100.trainer_metadata.json)
        latest_checkpoint(run_dir)                     -> .../checkpoint_100.trainer_metadata.json

    which would hand `resolve_checkpoint_dirs` a JSON file as `leaf_dir`. `trainer_metadata_
    <step>.json` does not match that glob (verified: `latest_checkpoint` is unchanged with one
    present, before and after a second save). Writing the record INSIDE the orbax leaf was the
    other candidate and is rejected for the same class of reason: the leaf is orbax's own
    directory format, and adding files to it is new, untested surface on the payload the loader
    reads.

    Pure string manipulation on the paths it is GIVEN; it resolves nothing and touches no disk."""
    return os.path.join(run_dir, f"trainer_metadata_{int(step)}.json")


def digest_pins_naming(path):
    """Every ``*.sha256`` file in ``path``'s directory whose CONTENT names ``path``'s basename —
    the hand-kept digest pins that a write to ``path`` would INVALIDATE.

    ONE definition, for both writers that can move a pinned document:
    ``tools/stamp_checkpoint_fingerprints.py`` (the retro-stamp) and
    ``experiments/832_linear_imprint/imprint.py::write_frame_provenance`` (the fingerprint
    refresh). The load-bearing provenance documents are pinned by hand —
    ``runs/861_ref/frame_provenance_ref.sha256`` covers all three
    ``frame_provenance_ref_s{0,1,2}.json`` — so changing their bytes invalidates a digest
    nothing in the tree recomputes. Neither caller rewrites those pins: a digest that a writer
    refreshes on its own behalf pins nothing. They NAME them, before writing.

    The mapping is not derivable from the filename (one digest file covers three documents), so
    it is read out of the digest files themselves — cheap, exact, and it cannot go stale.

    Read-only, and `stat`-before-`open` through `unsafe_to_open_defect` like every other sidecar
    read in this module (an unopenable/oversized candidate is skipped, never blocked on).
    Returns ABSOLUTE paths, sorted; callers relativize for display."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    base = os.path.basename(path)
    hits = []
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return hits
    for name in names:
        if not name.endswith(".sha256"):
            continue
        candidate = os.path.join(directory, name)
        if unsafe_to_open_defect(candidate, SIDECAR_MAX_BYTES,
                                 "a sha256 digest file") is not None:
            continue
        try:
            with open(candidate) as fh:
                if base in fh.read():
                    hits.append(candidate)
        except OSError:
            continue
    return hits


# The field a FRAME-PROVENANCE document uses to bind itself to the weights it describes. This
# is the name TASK_886's certification contract reads (`universe_pin["provenance_param_fingerprint"]`
# is sourced from it), so it lives here, beside the function that COMPUTES the value, rather
# than as a string literal in each writer/reader.
PROVENANCE_FINGERPRINT_FIELD = "param_fingerprint"

# TASK_894 scope 1 — the field `trainer.save_checkpoint` writes (into `trainer_metadata.json`
# AND its per-leaf `trainer_metadata_<step>.json` sibling) when the save happened under a
# PINNED frame, and the field `tools/stamp_checkpoint_fingerprints.py` retro-stamps from a
# checkpoint's own tracked frame-provenance document. `_certification_verdict`
# (`eval_vector.py`) reads it as the checkpoint's OWN INDEPENDENT record of the slot order it
# was written under — independent of the frame-provenance JSON document that sits beside it,
# because `param_fingerprint` (above) hashes parameter shape/dtype/bytes and has ZERO
# dependence on symbol names (measured, TASK_886: reversing `frame_symbols` in a checkpoint's
# own provenance with the fingerprint intact still certified). One name, defined once, so a
# save-time writer, the retro-stamp and the certification reader can never independently drift.
FRAME_SYMBOLS_SHA256_FIELD = "frame_symbols_sha256"


def provenance_path_for(ckpt_path):
    """``frame_provenance_<stem>.json`` beside ``ckpt_path`` (``ckpt_`` stripped off the
    basename) — the sibling file TASK_874's ``imprint.py`` writes next to every checkpoint it
    emits (``imprint.frame_provenance_path_for``).

    THE naming rule for the reader side, in one place: ``eval_vector._own_provenance_path_for``
    and ``tools/backfill_checkpoint_fingerprints.py`` both call this instead of keeping their
    own copy (``imprint.py`` keeps the writer-side copy because it imports FROM
    ``eval_vector``, and this module is imported by both). Carries imprint's documented caveat:
    the mapping is NOT injective — ``.../ckpt_foo`` and ``.../foo`` land on the same file.

    Pure string manipulation on the path it is GIVEN; it resolves nothing. Callers that want
    symlink-resolved identity pass a ``run_dir`` from `resolve_checkpoint_dirs` (the backfill
    does); callers preserving a pre-existing spelling pass their raw argv string."""
    base = os.path.basename(ckpt_path.rstrip(os.sep))
    stem = base[len("ckpt_"):] if base.startswith("ckpt_") else base
    return os.path.join(os.path.dirname(ckpt_path.rstrip(os.sep)) or ".",
                        f"frame_provenance_{stem}.json")


def sidecar_defect(run_dir):
    """`None` if every present run-dir sidecar is safe to open, else a human-readable defect
    phrase naming the offending one.

    WHY THIS EXISTS SEPARATELY FROM `unsafe_tree_defect` (fix-pass B2, strace-proven). The leaf
    walk covers everything orbax opens *under the leaf*. It does NOT cover
    `trainer_metadata.json` / `popart_state.npz`, which are SIBLINGS of the leaf: they sit in
    the RUN dir, one level up, and `trainer.load_checkpoint` opens both raw
    (`trainer.py`, inside `load_checkpoint`), as do `eval_vector.check_checkpoint_arch` and
    `tools/backfill_checkpoint_fingerprints.backfill_one`. A FIFO at either one therefore passed
    the leaf-only guard with NO defect and then blocked `openat` FOREVER — not a race, a
    100%-reproducible hang, because those paths were never walked at all.

    `stat` before `open`, via the same `unsafe_to_open_defect` every other sidecar read in this
    module uses. ABSENCE is not a defect: both sidecars are optional (older checkpoints carry
    neither, and `popart_state.npz` only exists when PopArt was enabled)."""
    for name in RUN_DIR_SIDECARS:
        path = os.path.join(run_dir, name)
        if not os.path.lexists(path):
            continue
        defect = unsafe_to_open_defect(path, SIDECAR_MAX_BYTES, f"a checkpoint's {name}")
        if defect is not None:
            return (f"its sidecar {name!r} (beside the run dir, which the loader opens raw) "
                    f"is unusable: {defect}")
    return None


# Bounds for the recursive leaf walk. Every real orbax checkpoint in this repo is 17-42 entries
# at depth <= 3, so these carry ~3 orders of magnitude of headroom. They do NOT bless a size —
# they exist so a corrupt/hostile tree (a symlink loop, a pathological fan-out) cannot turn the
# guard itself into the unbounded walk it is protecting against. Exceeding either is a REFUSAL,
# not a truncation: a partially-walked tree is not a verdict.
_LEAF_WALK_MAX_ENTRIES = 20_000
_LEAF_WALK_MAX_DEPTH = 32


def unsafe_tree_defect(root):
    """`None` if EVERY entry anywhere under `root` is a READABLE regular file (or a directory of
    them), else a human-readable defect phrase naming the offending entry relative to `root`.

    WHY A WALK AND NOT A FIXED FILE LIST. `trainer.load_checkpoint` -> `restore_checkpoint` ->
    orbax opens the PARAMETER PAYLOAD, not just the three marker files
    (`checkpoint`/`_METADATA`/`manifest.ocdbt`) and the two sidecars
    (`trainer_metadata.json`/`popart_state.npz`). Three strace-proven hangs on a copy of a
    committed smoke checkpoint, one file swapped for a FIFO each time, all past a marker-only
    guard and into an `openat` that never returned:
      * `checkpoint_<N>/d/<hash>`
      * `checkpoint_<N>/ocdbt.process_0/d/<hash>` (the multi-MB shard)
      * `checkpoint_<N>/array_metadatas/process_0`
    So this walks the WHOLE subtree the loader may read, bounded and loop-proof, by the same
    `stat` -> `S_ISREG` -> `os.access(R_OK)` idiom `unsafe_to_open_defect` uses for single
    sidecars. No size cap here: a parameter payload is legitimately hundreds of MB, and size is
    not what hangs a reader — type is.

    TYPE IS NOT THE ONLY WAY AN OPEN FAILS (fix-pass M4). A `chmod 000` payload file inside an
    otherwise genuine leaf is a regular file of plausible size, so it passed every check here and
    then failed from inside orbax/tensorstore as a bare `ValueError` — NOT a
    `CheckpointRefusalError` — which silently un-uniformed the refusal for callers that catch
    only this module's own exception type. Readability is therefore checked alongside type.

    BOUNDED BY CONSTRUCTION (else the guard would be the same defect one level up):
      * `os.stat` only — nothing is OPENED, and `stat` on a FIFO/socket/device returns
        immediately where `open` would block forever;
      * an explicit entry budget and depth cap (`_LEAF_WALK_MAX_ENTRIES` / `_LEAF_WALK_MAX_DEPTH`),
        both REFUSALS when exceeded;
      * directories are de-duplicated by `(st_dev, st_ino)`, so a symlink loop is walked once;
      * EVERY entry whose realpath escapes `root` is REFUSED — files as well as directories
        (fix-pass M2: the containment check used to run only on the `S_ISDIR` branch, so a
        directory-typed escape was correctly refused while a symlink to a regular file ANYWHERE
        on the host was blessed unconditionally. A blessed out-of-tree file is a checkpoint
        whose payload this guard cannot bound, and a file it never proved is the one the loader
        reads).

    WHAT THIS DOES NOT DO — stated precisely, because an overstated guard is worse than none.

    * SCOPE. It covers the LEAF SUBTREE only — everything at or under `root`. It says nothing
      about the RUN dir's sidecars (`trainer_metadata.json`, `popart_state.npz`), which are
      SIBLINGS of the leaf and are opened raw by `trainer.load_checkpoint`. Those are
      `sidecar_defect`'s job, and BOTH are called by `preflight_checkpoint_loadable` and
      `load_checkpoint_into`. This split is named because getting it wrong once already cost a
      reproducible hang: a leaf-only walk left the two sidecars uncovered, and that was not a
      timing race — it was 100% reproducible, because those paths were never walked.
    * TIME. Within its scope it proves every path was a readable regular file AT THE MOMENT IT
      WAS STAT'ED, before the loader ran. It does NOT hold a lock and cannot stop a file from
      BECOMING a FIFO/device, or from having its mode changed, between this check and
      `trainer.load_checkpoint`'s own `open`. THAT is the genuine TOCTOU remainder — a race,
      requiring a writer to act inside the window — and it is all that is left once the scope
      hole above is closed.
    * EUID. `os.access` answers for the process's CURRENT effective uid; running as root makes
      the readability half vacuously true. That weakens this check to "no worse than before" in
      that case, never to a false PASS of a type defect."""
    root_real = os.path.realpath(root)
    try:
        rst = os.stat(root_real)
    except OSError as e:
        return f"it cannot be stat'ed ({type(e).__name__}: {clip_for_refusal(e, quote=False)})"
    seen = {(rst.st_dev, rst.st_ino)}
    stack = [(root_real, 0)]
    n_entries = 0
    while stack:
        d, depth = stack.pop()
        try:
            # The budget is spent DURING the listing, not after: materialising an unbounded
            # directory and only then declaring it too big would be the unbounded read this
            # walk exists to refuse.
            entries = []
            with os.scandir(d) as it:
                for ent in it:
                    n_entries += 1
                    if n_entries > _LEAF_WALK_MAX_ENTRIES:
                        return (f"it holds more than {_LEAF_WALK_MAX_ENTRIES} entries — "
                                f"refusing to keep walking (a real checkpoint here is tens of "
                                f"files; an unbounded walk inside the guard would be the very "
                                f"hang this guard exists to prevent)")
                    entries.append(ent)
            entries.sort(key=lambda e: e.name)
        except OSError as e:
            return (f"its directory {clip_for_refusal(os.path.relpath(d, root_real))} cannot be "
                    f"listed ({type(e).__name__}: {clip_for_refusal(e, quote=False)}) — the "
                    f"loader reads the parameter payload from there")
        for ent in entries:
            rel = os.path.relpath(ent.path, root_real)
            try:
                st = os.stat(ent.path)        # FOLLOWS symlinks; stat never blocks, open does
            except OSError as e:
                return (f"its entry {clip_for_refusal(rel)} is present but cannot be stat'ed "
                        f"({type(e).__name__}: {clip_for_refusal(e, quote=False)}) — a dangling "
                        f"symlink / unreadable entry under the checkpoint is something the "
                        f"loader will try to open")
            # M2: containment FIRST, for every entry type. A symlinked-in regular file is just
            # as much "a path the loader opens that this guard cannot bound" as a symlinked-in
            # directory; typing the check to directories only meant the file case was blessed.
            child_real = os.path.realpath(ent.path)
            if not (child_real == root_real or child_real.startswith(root_real + os.sep)):
                return (f"its entry {clip_for_refusal(rel)} resolves OUTSIDE the checkpoint (to "
                        f"{clip_for_refusal(child_real)}) — refusing both to walk out of the "
                        f"leaf and to hand over a checkpoint whose payload this guard therefore "
                        f"cannot bound")
            if stat.S_ISREG(st.st_mode):
                # M4: type says the open will not BLOCK; it does not say it will SUCCEED. An
                # unreadable payload file (e.g. a chmod 000 `manifest.ocdbt`) passed every guard
                # here and then failed inside orbax/tensorstore as a bare ValueError, so a
                # caller catching only CheckpointRefusalError saw an untyped crash instead of
                # this module's named refusal. No size cap: a payload is legitimately huge.
                if not os.access(ent.path, os.R_OK):
                    return (f"its entry {clip_for_refusal(rel)} is a regular file but is NOT "
                            f"READABLE by this process — refusing to hand this checkpoint to "
                            f"the loader: orbax opens the parameter payload under this "
                            f"directory, and an unreadable file there surfaces as an untyped "
                            f"error from inside the reading library, not a refusal here")
                continue
            if not stat.S_ISDIR(st.st_mode):
                return (f"its entry {clip_for_refusal(rel)} is NOT a regular file "
                        f"(FIFO/socket/device) — refusing to hand this checkpoint to the "
                        f"loader: orbax opens the parameter payload under this directory, and "
                        f"opening a FIFO or device there blocks FOREVER, which is a hang, not a "
                        f"verdict")
            key = (st.st_dev, st.st_ino)
            if key in seen:
                continue                      # symlink/bind loop — walked once, never again
            seen.add(key)
            if depth + 1 > _LEAF_WALK_MAX_DEPTH:
                return (f"its subdirectory {clip_for_refusal(rel)} is nested deeper than "
                        f"{_LEAF_WALK_MAX_DEPTH} levels — refusing to keep walking")
            stack.append((child_real, depth + 1))
    return None


def leaf_escape_defect(run_dir, leaf):
    """`None` if `leaf` really lives inside `run_dir`, else a human-readable defect phrase.

    Fix-pass MINOR: `flax_checkpoints.latest_checkpoint` builds its answer as
    `os.path.join(ckpt_dir, <bare basename from os.listdir>)` — it never realpaths the FINAL
    component. So a `checkpoint_0` SYMLINK planted inside run dir A, pointing at a well-formed
    checkpoint anywhere else on the host, is auto-discovered, passes every other guard (the tree
    walk realpaths its own root, so the foreign payload is walked and found clean) and is loaded.
    Nothing hangs and the fingerprint stays accurate — this is NOT a fail-open — but the run dir
    the guard resolved and the bytes the loader read are then two different checkpoints, which
    is this row's guard/loader identity thesis one layer deeper. The sidecars compound it:
    `trainer_metadata.json` / `popart_state.npz` are read from `run_dir` (A) while the parameters
    come from elsewhere (B), so an arch check can pass on A's metadata over B's weights.

    ONLY BITES ON AUTO-DISCOVERY, by construction rather than by a flag: when argv names the
    orbax leaf directly, `resolve_checkpoint_dirs` returns `run_dir = dirname(realpath(leaf))`,
    so containment holds trivially and this can never fire. An operator who explicitly points
    `--checkpoint` at a leaf has NAMED the bytes they want, wherever they live; a leaf this
    program picked FOR them by listing a directory must be the one that directory actually
    holds."""
    leaf_real = os.path.realpath(leaf)
    root_real = os.path.realpath(run_dir)
    if leaf_real == root_real or leaf_real.startswith(root_real + os.sep):
        return None
    return (f"the checkpoint leaf auto-discovered inside it resolves OUTSIDE that directory "
            f"(to {clip_for_refusal(leaf_real)}) — refusing to load a payload from one "
            f"checkpoint while reading the run dir's own trainer_metadata.json / "
            f"popart_state.npz sidecars from another. Point --checkpoint at the leaf you "
            f"actually mean if that is what you want")


def checkpoint_leaf_defect(leaf):
    """`None` if `leaf` is a checkpoint `restore_checkpoint` can actually READ, else a
    human-readable defect phrase.

    flax's `_is_orbax_checkpoint` decides purely with `os.path.exists`: it never asks WHAT the
    thing it found is. Two measured consequences:

    * `mkdir ckpt_evil && mkfifo ckpt_evil/checkpoint` reports LOADABLE, and the run then blocks
      FOREVER in `openat(".../checkpoint", O_RDONLY)` (strace-proven). A hang is not a verdict.
    * a `checkpoint_<N>` child that merely MATCHES the name pattern but is an EMPTY directory
      passes `latest_checkpoint` (a listdir + glob, nothing more), and flax then falls through to
      its legacy msgpack path and `open()`s the directory — `IsADirectoryError`, raised from
      inside flax.

    The marker scan below is not the whole checkpoint: orbax opens the parameter payload too
    (`d/<hash>`, `ocdbt.process_0/d/<hash>`, `array_metadatas/process_0`), and a FIFO at any of
    those hangs the run past a marker-only guard (three strace-proven reproductions). So after
    the marker checks (kept ahead of the walk only so their specific wording survives),
    `unsafe_tree_defect` walks the ENTIRE leaf subtree, bounded and loop-proof.

    SCOPE OF THE CLAIM, precisely. Read-only, opens nothing: `lstat` for the leaf, `stat` +
    `os.access(R_OK)` for every entry beneath it. It establishes that at the moment of the walk —
    before `trainer.load_checkpoint` runs — every path the loader can open UNDER THIS LEAF was a
    regular file THIS PROCESS COULD READ. It says nothing about the RUN dir's sidecars, which the
    loader also opens raw and which live one level up: that is `sidecar_defect`'s half, and both
    callers (`preflight_checkpoint_loadable`, `load_checkpoint_into`) run BOTH. It does NOT
    re-check anything at open time (see `unsafe_tree_defect`'s disclosed TOCTOU/euid remainders).
    A leaf that is itself a regular file is flax's legacy single-file msgpack checkpoint and is
    safe to hand over once readable (a regular file cannot block the reader; an unreadable one
    still cannot be opened)."""
    try:
        st = os.lstat(leaf)
    except OSError as e:
        return f"it cannot be stat'ed ({type(e).__name__}: {clip_for_refusal(e, quote=False)})"
    if stat.S_ISLNK(st.st_mode):                      # follow once: realpath already did for
        try:                                          # the argv path, not for a listdir'ed leaf
            st = os.stat(leaf)
        except OSError as e:
            return (f"it is a symlink that cannot be resolved "
                    f"({type(e).__name__}: {clip_for_refusal(e, quote=False)})")
    if stat.S_ISREG(st.st_mode):                      # flax legacy msgpack file
        if not os.access(leaf, os.R_OK):              # M4: safe to open != possible to open
            return ("it is a legacy single-file checkpoint but is NOT READABLE by this "
                    "process — refusing to hand it to the loader")
        return None
    if not stat.S_ISDIR(st.st_mode):
        return ("it is NOT a regular file or a directory (FIFO/socket/device) — refusing to "
                "hand it to the loader: opening a FIFO or device here blocks FOREVER, which is "
                "a hang, not a verdict")
    present = []
    for marker in _ORBAX_MARKERS:
        mpath = os.path.join(leaf, marker)
        if not os.path.lexists(mpath):
            continue
        try:
            mst = os.stat(mpath)
        except OSError as e:
            return (f"its orbax marker {marker!r} is present but cannot be stat'ed "
                    f"({type(e).__name__}: {clip_for_refusal(e, quote=False)})")
        if not stat.S_ISREG(mst.st_mode):
            return (f"its orbax marker {marker!r} is present but is NOT a regular file "
                    f"(FIFO/socket/device/directory) — refusing to hand it to the loader: "
                    f"opening a FIFO or device here blocks FOREVER, which is a hang, not a "
                    f"verdict")
        if not os.access(mpath, os.R_OK):
            return (f"its orbax marker {marker!r} is present and a regular file but is NOT "
                    f"READABLE by this process — refusing to hand it to the loader: the open "
                    f"fails inside orbax/tensorstore as an untyped error, not a refusal here")
        present.append(marker)
    if not present:
        return ("it is a directory matching the 'checkpoint_<step>' name pattern but carries "
                f"NONE of orbax's marker files ({', '.join(map(repr, _ORBAX_MARKERS))}) — flax "
                f"would fall through to its legacy single-file path and open() a directory, "
                f"raising IsADirectoryError from inside the loader")
    return unsafe_tree_defect(leaf)


def preflight_checkpoint_loadable(ckpts):
    """Refuse a nonexistent/unloadable `--checkpoint` BEFORE any panel/model is built (the
    cheapest, highest-severity check in this module: a directory-listing probe, no model build,
    no load). Resolves each checkpoint via `resolve_checkpoint_dirs` — the SAME function the
    loader uses — never a re-implementation that can drift from it.

    This is the FAST half of the fix; `load_checkpoint_into`'s object-identity check is the
    load-time half — it protects callers that never run this preflight, and is what actually
    produces the load-evidence fingerprint. Both must hold; this one exists so the refusal is
    CHEAP and fires before any panel/model cost is paid.

    One poisoned entry in a list ABORTS THE WHOLE CALL, including on behalf of honest entries
    ordered after it. That is deliberate and is the conservative direction: a batch that silently
    drops members produces a comparison over an unstated subset. Re-run the list without the
    named entry to grade the rest.

    Covers BOTH halves of what the loader opens — the leaf subtree (`checkpoint_leaf_defect` ->
    `unsafe_tree_defect`) and the run dir's raw-opened sidecars (`sidecar_defect`) — plus
    `leaf_escape_defect`, which binds an AUTO-DISCOVERED leaf to the run dir it was discovered
    in (flax's `latest_checkpoint` joins a bare `os.listdir` basename without realpathing it).

    CORRECTOR: refuses with the operator's `--checkpoint` strings untouched; never guesses,
    substitutes or repairs a path (the ECHO of those strings is clipped — see
    `clip_for_refusal` — which changes how the refusal is spelled, never what it refuses)."""
    for path in ckpts:
        realpath = os.path.realpath(path)
        run_dir, leaf_dir = resolve_checkpoint_dirs(path)
        if leaf_dir is None:
            exists_note = (
                "the path exists but flax finds no orbax checkpoint there and no "
                "'checkpoint_<step>' child inside it" if os.path.exists(realpath)
                else "the path does not exist")
            raise RuntimeError(
                f"--checkpoint: REFUSING — no loadable checkpoint found at "
                f"{clip_for_refusal(path)} (resolved: {clip_for_refusal(realpath)}; "
                f"{exists_note}).\n"
                f"  Without this check, flax's restore_checkpoint silently returns the "
                f"trainer's UNCHANGED random-init state for a nonexistent/empty checkpoint "
                f"dir instead of raising (TASK_888/D37) — this run would otherwise grade or "
                f"trade random-init weights under a report that says otherwise.\n"
                f"  FIX: point --checkpoint at a real checkpoint (a run dir containing "
                f"checkpoint_<step>/, or that orbax leaf directly).")
        defect = (leaf_escape_defect(run_dir, leaf_dir) or checkpoint_leaf_defect(leaf_dir)
                  or sidecar_defect(run_dir))
        if defect is not None:
            raise RuntimeError(
                f"--checkpoint: REFUSING — the checkpoint found at {clip_for_refusal(path)} is "
                f"NOT LOADABLE.\n"
                f"  resolved leaf: {clip_for_refusal(leaf_dir)}\n"
                f"  what is wrong: {defect}\n"
                f"  FIX: point --checkpoint at a real checkpoint (a run dir containing a "
                f"populated checkpoint_<step>/, or that orbax leaf directly).")


# =============================================================================
# load-time evidence + fail-closed load (scope 2/3)
# =============================================================================

def loaded_param_shape_defect(before, after, max_named=6):
    """`None` if every restored leaf has EXACTLY the shape the freshly-built model had, else a
    `(n_total, n_changed, detail)` record — the signal that names an `--n-assets`/architecture
    mismatch.

    A checkpoint trained on a DIFFERENT asset count restores with FOREIGN SHAPES: flax fills
    every leaf whose name it finds, so per-asset-sized leaves come back at the CHECKPOINT's
    shape while the rest keep the model's. Because a real restore genuinely happened, neither
    the object-identity no-op detector nor an architecture guard keyed on
    `trainer_metadata.json` (which records no asset count) has anything to say — the run would
    otherwise die several frames later inside flax with a bare `ScopeParamShapeError`, after
    every earlier guard reported nothing wrong.

    EXACTLY WHAT THIS CAN AND CANNOT SEE (fix-pass B4 — the previous docstring's "A PARTIAL
    restore is refused too" was an OVERSTATEMENT, and in this program an overstated guard is
    itself the defect). `restore_checkpoint(target=...)` finishes with
    `serialization.from_state_dict(target, restored)`, which returns a pytree CONGRUENT TO THE
    TARGET: same leaf paths, always. So `before` and `after` have IDENTICAL key sets by
    construction, and this comparison can only ever speak about SHAPES:

      * DETECTED — a leaf restored at a shape the model did not build (the `--n-assets` /
        model-size mismatch this exists for), because the foreign shape survives into the
        returned tree.
      * NOT DETECTED — a leaf the model has that the checkpoint LACKS (silently keeps the
        model's fresh-init value), or a leaf the checkpoint has that the MODEL lacks (silently
        dropped). Both are shape-invisible: the key sets cannot differ. The set-difference
        branches this function used to carry were therefore UNREACHABLE and have been removed
        rather than left standing as decoration for a claim they never checked. Measured
        example: a model built OUTSIDE this program's canonical
        `TrainingConfig` -> `setup_model_and_trainer` path lacked `vector_action_embedding`
        entirely, and a real checkpoint loaded into it reported `"loaded": true` with a stable
        fingerprint while dropping `vec_act_proj.{kernel,bias}`. The countermeasure is that
        every load site builds through the CANONICAL path (so the target IS the architecture),
        plus `eval_vector.check_checkpoint_arch`'s h/l-cycles + state_dim guard — not this
        function.

    Shapes only — no values are read, no device transfer is forced.

    CORRECTOR: reports; the caller refuses and leaves `trainer.state` untouched."""
    def _shapes(tree):
        return {jax.tree_util.keystr(p): tuple(np.shape(v))
                for p, v in jax.tree_util.tree_flatten_with_path(tree)[0]}
    b, a = _shapes(before), _shapes(after)
    changed = sorted(k for k in (b.keys() & a.keys()) if b[k] != a[k])
    if not changed:
        return None
    lines = [f"    {k}: model built {b[k]}, checkpoint restored {a[k]}"
             for k in changed[:max_named]]
    if len(changed) > max_named:
        lines.append(f"    ... and {len(changed) - max_named} more leaf/leaves with a "
                     f"different shape (names not all echoed)")
    return (len(b), len(changed), "\n".join(lines))


def param_fingerprint(params):
    """Deterministic sha256 fingerprint of a params pytree — the load/save evidence this whole
    module exists to produce (TASK_888 scope 3/4).

    Leaves are sorted by their tree PATH string (`jax.tree_util.tree_flatten_with_path` +
    `jax.tree_util.keystr`), NOT by dict/insertion order — flax params are nested dicts, and dict
    iteration order being stable within one process for one construction sequence is not a
    promise a cross-process fingerprint should lean on. Sorting by path string is a total order
    independent of how any upstream dict happened to build its keys.

    Read-only: hashes shape+dtype+bytes per leaf in that sorted order; never mutates or repairs
    the pytree it is given. Verified (see `test_checkpoint_io.py`): the SAME checkpoint loaded
    twice fingerprints identically; a DIFFERENTLY-trained checkpoint (same architecture)
    fingerprints differently."""
    leaves_with_path = jax.tree_util.tree_flatten_with_path(params)[0]
    keyed = sorted(
        ((jax.tree_util.keystr(p), np.asarray(v)) for p, v in leaves_with_path),
        key=lambda kv: kv[0])
    h = hashlib.sha256()
    for key, arr in keyed:
        h.update(key.encode("utf-8"))
        h.update(repr(arr.shape).encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


# ASCII Unit Separator (0x1F) — a control character no ticker symbol observed anywhere in this
# repo's asset tables contains. A printable separator (","/"|"/" ") could turn two DIFFERENT
# ordered lists into the SAME joined string if a symbol ever contained it (e.g. ["AB", "C"] and
# ["A", "BC"] joined on "" collide; even "," is not obviously safe against a future asset name).
# Fixed here so a save-time writer and a certification-time reader can never independently pick
# a different separator and silently disagree about what the digest means.
_FRAME_SYMBOLS_DIGEST_SEP = "\x1f"


def frame_symbols_digest(symbols):
    """Deterministic sha256 over an ORDERED symbol list (TASK_894 scope 1).

    THE digest `trainer.save_checkpoint` records (as `FRAME_SYMBOLS_SHA256_FIELD`) when a save
    happens under a pinned frame, and the one `tools/stamp_checkpoint_fingerprints.py` derives
    from a checkpoint's own tracked frame-provenance document when retro-stamping. Reused,
    never re-derived, by `eval_vector.py`'s certification predicate, so all three sites agree on
    what "the digest of this ordered list" means.

    CANONICAL ENCODING (documented once, here). Each symbol, in the order given — never sorted;
    sorting would destroy the very thing this digest exists to protect — UTF-8 encoded, joined
    by a single `_FRAME_SYMBOLS_DIGEST_SEP` byte with no trailing separator and no JSON
    wrapping. `str(s)` normalises any non-string leaf (there should never be one) into the same
    byte sequence a JSON string would encode to, rather than raising deep inside `.encode()`.

    Read-only: never mutates, sorts or reorders `symbols`. `list(symbols)` twice on the same
    ordered input always fingerprints identically; a reversed or otherwise reordered list never
    does (this is the whole point — `param_fingerprint` above cannot see order at all)."""
    blob = _FRAME_SYMBOLS_DIGEST_SEP.join(str(s) for s in symbols)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class CheckpointRefusalError(RuntimeError):
    """Raised by `load_checkpoint_into` on every fail-closed path. A plain subclass of
    `RuntimeError` (every caller already catches/propagates `RuntimeError`); named so a caller
    that wants to distinguish "this checkpoint is bad" from an unrelated `RuntimeError` can."""


def load_checkpoint_into(trainer, ckpt_path, *, allow_fresh_init=False,
                         allow_architecture_mismatch=False):
    """THE loader every one of the four TASK_888 sites calls instead of a bare
    `trainer.load_checkpoint(...)` — resolves via `resolve_checkpoint_dirs` (the SAME function
    the guard uses), then REFUSES rather than continues on anything that is not a real, complete
    load. Mutates `trainer.state` in place on success and returns a load-evidence dict; on a
    permitted fresh-init (see below) `trainer.state` is left exactly as the caller built it.

    ``allow_architecture_mismatch`` (TASK_889 / D39, default False): threaded straight through
    to `trainer.load_checkpoint`, which REFUSES (before this function ever mutates
    `trainer.state`) when the checkpoint's recorded `vector_action`/`use_prediction_head`
    disagrees with the target's own, or its recorded `param_fingerprint` disagrees with what
    this load actually restored. Pass True only from a call site that offers its OWN explicit
    flag for "yes, I mean to load a mismatched checkpoint" — the same discipline
    `allow_fresh_init` already uses here.

    Returns ``{"resolved_path": str, "step": int, "param_fingerprint": str, "loaded": bool}``.
    `resolved_path` and `param_fingerprint` describe whatever params end up live in
    `trainer.state` after this call returns — i.e. they are ALWAYS populated, whether that is a
    freshly-loaded checkpoint or (only with `allow_fresh_init=True`) the untouched fresh init the
    caller built before calling this. This is the load-time evidence TASK_888 scope 3 asks for:
    independent of provenance, it makes two runs of the "same" checkpoint comparable and detects
    a checkpoint changing underneath you.

    A LOAD THAT COULD NOT BE VERIFIED SAYS SO, IN THE EVIDENCE (TASK_889 fix pass). ``loaded:
    True`` used to be indistinguishable from a VERIFIED load at all five call sites: a checkpoint
    whose recorded architecture/fingerprint could not be compared (nothing was ever recorded, or
    the caller passed ``allow_architecture_mismatch``) returned the identical dict as one whose
    every recorded field was checked and agreed. Three extra keys —
    ``architecture_verified`` / ``fingerprint_verified`` / ``verification_skipped`` (a
    human-readable reason) — are therefore merged in from `trainer.load_checkpoint`'s own record
    of what it was able to check.

    THEY APPEAR ONLY IN THE DEGRADED CASE, deliberately, and that is not decoration: this dict is
    embedded VERBATIM in a graded artifact (``eval_vector.py``'s ``checkpoint_load`` block) and
    that artifact is compared BYTE-FOR-BYTE against a committed baseline by
    ``861/smoke_e2e.py`` leg (i) — every key except ``resolved_path`` is pinned. Unconditional
    new keys would turn a green leg red for a fully-verified load and force a planner-gated
    baseline promotion to say nothing new. A field present only in the degraded case is this
    repo's existing idiom for exactly this (TASK_883's ``universe_pinned: false``, TASK_874's
    ``provenance_replaced``): ``evidence.get("verification_skipped")`` is the machine-visible
    test, None/absent = every recorded field was compared and agreed.

    ``allow_fresh_init`` (default False): when nothing loadable exists at `ckpt_path`, the
    default is a hard refusal (scope 2 — "never a fallback"). Pass True only from a call site
    that offers its OWN explicit flag for "yes, I mean to run on fresh-init weights" (e.g.
    `distill.py --allow-fresh-init`); real_trade_jax.py and train_jax_fast.py's
    `--load-checkpoint` never pass it, because there is no legitimate reason for either to
    silently trade or resume on random-init weights.

    THE DETECTOR FOR "DID ANYTHING ACTUALLY LOAD" IS OBJECT IDENTITY, NOT NUMERIC EQUALITY. A
    no-op load returns *the very same object*: `loaded is state_before` and `loaded.params is
    params_before` are both True on a miss, both False on a real load (flax rebuilds the pytree
    through `from_state_dict`, so every leaf is a fresh array even when the VALUES are
    identical). Numeric equality is unsound here: a checkpoint saved straight from a fresh-init
    model is NUMERICALLY equal to another fresh init with the same seed, and this program
    deliberately ships/tests such checkpoints — a numeric-equality detector refuses an honest
    load of one (measured: it broke `861/smoke_e2e` and `835/scorecard_smoke` in an earlier
    cycle of this fix).

    A FOREIGN-SHAPE restore is refused too (`loaded_param_shape_defect`): object identity
    answers "did anything load", not "does what loaded FIT" — a checkpoint trained on a
    different `--n-assets`/architecture restores per-asset leaves at the CHECKPOINT's shape,
    which is a REAL load (identity correctly does not fire) that dies several frames later
    inside flax with a bare `ScopeParamShapeError` if let through. Read that check's docstring
    for what it can and cannot see; it is a shape check, not a leaf-set check.

    THE EVIDENCE AND THE LOAD DESCRIBE THE SAME BYTES (fix-pass B1). The resolved LEAF is what
    is handed to `trainer.load_checkpoint`, not the run dir. Passing the run dir made the loader
    re-decide which checkpoint to open — `restore_checkpoint(step=None)` takes
    `latest_checkpoint(run_dir)` — so with more than one `checkpoint_<step>` in a run dir (the
    DEFAULT: `save_checkpoint(keep=3)`), naming an OLDER leaf explicitly had the guard bless
    leaf A while the loader installed leaf B, and the returned evidence reported
    `resolved_path` = A beside a `step`/`param_fingerprint` = B. That is this row's own thesis
    ("argv is not evidence") re-opened one layer inside the resolver built to close it.

    CORRECTOR: refuses on a fail-open load; never repairs, retries or substitutes a path — on
    every refusal `trainer.state` is left exactly as the caller built it."""
    run_dir, leaf_dir = resolve_checkpoint_dirs(ckpt_path)
    if leaf_dir is None:
        if allow_fresh_init:
            state = trainer.state
            return {
                "resolved_path": run_dir,
                "step": int(np.asarray(state.step_count)),
                "param_fingerprint": param_fingerprint(state.params),
                "loaded": False,
            }
        exists_note = ("the path exists but no orbax checkpoint / 'checkpoint_<step>' child "
                       "was found there" if os.path.exists(run_dir) else "the path does not exist")
        raise CheckpointRefusalError(
            f"REFUSING checkpoint {clip_for_refusal(ckpt_path)}: no loadable checkpoint found "
            f"(resolved: {clip_for_refusal(run_dir)}; {exists_note}).\n"
            f"  Without this refusal, flax's restore_checkpoint silently returns the trainer's "
            f"UNCHANGED random-init state for a nonexistent/empty checkpoint dir instead of "
            f"raising (TASK_888/D37/D38) — this run would otherwise continue on random-init "
            f"weights under a report that says the checkpoint loaded.\n"
            f"  FIX: point at a real checkpoint (a run dir containing checkpoint_<step>/, or "
            f"that leaf directly).")
    defect = (leaf_escape_defect(run_dir, leaf_dir) or checkpoint_leaf_defect(leaf_dir)
              or sidecar_defect(run_dir))
    if defect is not None:
        raise CheckpointRefusalError(
            f"REFUSING checkpoint {clip_for_refusal(ckpt_path)}: not loadable — resolved leaf "
            f"{clip_for_refusal(leaf_dir)} — {defect}.\n"
            f"  FIX: point at a real checkpoint (a run dir containing a populated "
            f"checkpoint_<step>/, or that orbax leaf directly).")
    state_before = trainer.state
    params_before = state_before.params
    # `leaf_dir` — the EXACT directory the guard above verified — is what gets restored; the run
    # dir goes along only because the two sidecars live beside it (B1, see the docstring).
    loaded = trainer.load_checkpoint(run_dir, leaf_dir=leaf_dir,
                                     allow_architecture_mismatch=allow_architecture_mismatch)
    if loaded is None:
        raise CheckpointRefusalError(
            f"trainer.load_checkpoint returned None for resolved leaf "
            f"{clip_for_refusal(leaf_dir)} (--checkpoint argument was "
            f"{clip_for_refusal(ckpt_path)})")
    if loaded is state_before or loaded.params is params_before:
        raise CheckpointRefusalError(
            f"REFUSING checkpoint {clip_for_refusal(ckpt_path)}: trainer.load_checkpoint "
            f"returned the trainer's OWN state object back, unchanged (resolved leaf: "
            f"{clip_for_refusal(leaf_dir)}) — flax's "
            f"restore_checkpoint silently returns its `target` argument AS THE SAME OBJECT when "
            f"it finds nothing loadable at this path, which is neither a raise nor a None "
            f"return (TASK_888/D37), so this run would otherwise continue on random-init "
            f"weights as if they were the checkpoint's. (A real load always rebuilds the "
            f"pytree, so this can NEVER fire on a checkpoint that was actually read — including "
            f"one whose weights happen to equal this trainer's init.)")
    shape_defect = loaded_param_shape_defect(params_before, loaded.params)
    if shape_defect is not None:
        n_total, n_changed, detail = shape_defect
        raise CheckpointRefusalError(
            f"REFUSING checkpoint {clip_for_refusal(ckpt_path)} (resolved leaf: "
            f"{clip_for_refusal(leaf_dir)}): it restored FOREIGN SHAPES into this model — "
            f"{n_changed} of {n_total} parameter leaves came back at the CHECKPOINT's shape "
            f"instead of the model's.\n{detail}\n"
            f"  The commonest cause is a --n-assets (or model-size/h-cycles/l-cycles) that does "
            f"not match the run this checkpoint came from.")
    trainer.state = loaded
    evidence = {
        "resolved_path": leaf_dir,
        "step": int(np.asarray(loaded.step_count)),
        "param_fingerprint": param_fingerprint(loaded.params),
        "loaded": True,
    }
    # TASK_889 fix pass: what `trainer.load_checkpoint` was actually able to VERIFY, merged in
    # only when something could not be (see the docstring for why "only when" is load-bearing —
    # the committed-artifact byte comparison in 861 leg (i)). `getattr` with a default because
    # the attribute is this loader's contract with `trainer.load_checkpoint`, and a trainer-like
    # object that predates it must degrade to the old dict rather than crash.
    verification = getattr(trainer, "last_load_verification", None)
    if isinstance(verification, dict) and verification.get("verification_skipped"):
        evidence.update(verification)
    return evidence
