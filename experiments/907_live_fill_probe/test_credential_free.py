"""Confirmation, structurally enforced: no `.py` file anywhere in this lane's tree reads/logs/
references any credential, and nothing here imports the private/authenticated Kraken REST path or
builds a real order-transmission call. Grepped rather than asserted in prose, so a future edit
that adds a credential reference — or a brand NEW file this fence has never heard of — fails CI
rather than fails silently.

D68/TASK_910 scope 3 (run 44's MINOR finding): the fence used to be a hardcoded FILENAME
allowlist (`PHASE1_FILES` + `ALL_SOURCE_FILES`). A curated list of names covers only the files
someone remembered to add to it — `__init__.py` was unscanned by construction, and the very
next new source file in this lane (this row shipped `health_probe.py` after the allowlist was
last edited) is born outside it until someone notices and appends its name. The fence is now a
recursive walk of every `.py` file under the lane directory — see `_lane_py_files` below.

WHAT THIS MECHANISM GUARANTEES, AND WHAT IT DOES NOT
----------------------------------------------------
This is a literal-substring grep, matched case-insensitively, over the lane's whole `.py` tree.

It DOES catch accidental, non-adversarial reintroduction: someone adding a credential read, an
authenticated-REST import, or an order-transmission call and spelling it the ordinary way — in
today's files, in a file that does not exist yet, or in a file at ANY depth below the lane root
(a `venue/order_client.py`-shaped phase-2 increment, which is precisely the case D68 names —
and equally `venue/build/order_client.py`, see the skip rules below). The ordinary aliasing
spellings are covered too (`from os import environ`, `getattr(os, ...)`, `__import__`,
`importlib`, and any capitalisation of the order methods).

WHAT IS SKIPPED, PRECISELY (do not read "any depth" as "unconditionally")
Two sets with DIFFERENT depth rules, because "skip everything named `build` anywhere" and "skip
nothing but the top level" are both wrong. `ALWAYS_SKIP` (`__pycache__`, `.pytest_cache`,
`.mypy_cache`, `.ruff_cache`, `.tox`, `.git`) is machine-generated and unambiguous, so it is
skipped at EVERY depth — a nested `venue/__pycache__/x.py` is not lane source. `TOP_LEVEL_ONLY_SKIP`
(`.venv`, `venv`, `env`, `trader_env`, `site-packages`, `node_modules`, `build`, `dist`) holds
names a real lane module could legitimately use, so it is skipped ONLY as a direct child of the
lane root, where it means "a vendored environment or a packaging output tree dropped in the lane".
`<lane>/build/x.py` is skipped; `<lane>/venue/build/order_client.py` is SCANNED. TASK_910 cycle-3
BLOCKER: matching those ambiguous names at every depth made any subdirectory whose name collided
with the set invisible, and `build/`/`dist/` are exactly what Python packaging produces — a full
duplicate of every source file, credential-reading code included.

SYMLINKS ARE A REFUSAL, NOT A SKIP
`Path.rglob` (and `os.walk` without `followlinks`) does not descend into a directory reached via
a symlink, so an ordinary refactor — `ln -s ../shared/venue_client.py venue/` — would import and
execute normally while never being scanned. Rather than document that hole, the walk DETECTS
symlinked directories and symlinked `.py` files under the lane and REFUSES: `_lane_py_files`
raises `FenceCannotVouchError` naming the symlink, so every scanning test here fails loudly. A
fence that cannot see something must say so, not stay green. Resolve the symlink into a real
file/directory in the lane (or move the target under the lane) to make the fence readable again.

It DOES NOT prove the absence of an order path against a DELIBERATE evader. A literal grep is
defeated by string assembly (`"os.env" + "iron"`, f-string token building), by base64/`exec`, by
a name bound at runtime from data, or by an order call that never spells an order word. Nothing
here searches the AST or the runtime; there is no taint analysis and no import-graph check.
D68's whole point was retiring docstring-strength safety claims, so this file states its own
strength honestly: it is a regression fence against forgetting, not a sandbox against intent.
"""
import os
import shutil
from pathlib import Path

import pytest

_THIS_DIR = Path(__file__).resolve().parent
_THIS_FILE = Path(__file__).resolve()

#: Machine-generated directories, skipped at ANY depth. Their names are unambiguous — no lane
#: module would ever be called `__pycache__` — so a nested `venue/__pycache__/x.py` is safely
#: assumed to be interpreter output, never lane source.
ALWAYS_SKIP = frozenset({
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".git",
})

#: Names a REAL lane module could legitimately use, so they are skipped ONLY when they sit
#: directly in the lane root — i.e. a vendored environment or a packaging output tree dropped in
#: the lane, not a subpackage. TASK_910 cycle-3 BLOCKER: skipping these at every depth reopened
#: the exact hole cycle 2 closed — `venue/build/order_client.py` reading credentials and building
#: an order frame passed the whole fence clean, and `build/`/`dist/` are precisely what Python
#: packaging emits (a full duplicate of every source file). Do NOT move a name from here into
#: `ALWAYS_SKIP` to quiet a slow walk; that trade is how the hole got in.
TOP_LEVEL_ONLY_SKIP = frozenset({
    ".venv", "venv", "env", "trader_env", "site-packages", "node_modules", "build", "dist",
})


class FenceCannotVouchError(RuntimeError):
    """The walk found something it cannot see behind (today: a symlink), so it refuses.

    Deliberately NOT an `AssertionError` subclass: the `pytest.raises(AssertionError)` positive
    controls below must not be able to pass on a refusal instead of on a real forbidden-token hit.
    """


def _dir_is_skipped(rel_parts):
    """Skip decision for one directory, given its path parts relative to the lane root."""
    name = rel_parts[-1]
    if name in ALWAYS_SKIP:
        return True
    return len(rel_parts) == 1 and name in TOP_LEVEL_ONLY_SKIP

#: Matched CASE-INSENSITIVELY as literal substrings (see `_forbidden_hits`). Case-insensitivity
#: is what makes `addOrder` / `ADD_ORDER` / `Add_Order` trip the same wire as `add_order`; it is
#: applied uniformly rather than per-token because every entry here is a symbol/attribute name
#: whose casing carries no meaning worth preserving, and no entry is short or word-like enough to
#: collide with English prose in a docstring (verified: the whole lane stays green).
#:
#: `os.environ`/`getenv` (any credential read, not just the two named Kraken env vars) and
#: `add_order`/`AddOrder` (the venue's real order-transmission call, in either spelling) were
#: added at D68/TASK_910: run 44 demonstrated a new file reading both credential env vars and
#: building an `add_order` frame passed the OLD fence (3 passed) because `FORBIDDEN` contained
#: no env-access token at all and the new file wasn't in `PHASE1_FILES`/`ALL_SOURCE_FILES`.
#: The aliasing spellings (`from os import environ`, `getattr(os`, `__import__`, `importlib`)
#: were added in TASK_910's cycle-2 fix pass: they are the ORDINARY ways to reach `os.environ`
#: without writing `os.environ`, and a fence that missed them while claiming to cover credential
#: reads would be overstating itself. They are cheap to catch literally; see the module docstring
#: for the spellings that remain out of reach of any grep.
#:
#: THE FALSE POSITIVES ARE DELIBERATE — READ THIS BEFORE "FIXING" ONE (TASK_910 cycle-3 MINOR).
#: Uniform case-insensitive substring matching means plausible identifiers trip this fence even
#: when they are innocent: `AddOrderBook`, `add_order_level`, `importlib` merely NAMED in a
#: comment. That is not a defect. This lane genuinely contains order-book-shaped code
#: (`test_book_normalization.py`, `shadow_engine.py`), so the noise is the price of erring in the
#: fail-closed direction, which is the correct direction for a fence: a false RED costs one
#: rename, a false GREEN costs the invariant. The correct response to a false positive is to
#: RENAME the identifier, or — if it truly cannot be renamed — to add a narrowly-scoped,
#: individually-reviewed exception at that one site with the reasoning written down. It is NOT to
#: narrow the token set, drop case-insensitivity, or reintroduce a per-file allowlist: the
#: per-file allowlist is exactly the cycle-1 defect D68 was raised to kill, and every "just make
#: this one token less noisy" edit rebuilds it a token at a time.
FORBIDDEN = (
    # credential names / authenticated-REST machinery
    "KRAKEN_API_KEY", "KRAKEN_API_SECRET", "_private_request", "api_secret", "api_key",
    # environment access, direct and via the ordinary aliases
    "os.environ", "getenv", "from os import environ", "getattr(os", "__import__", "importlib",
    # the venue's real order-transmission call
    "add_order", "AddOrder",
)


def _forbidden_hits(text):
    """Every `FORBIDDEN` token literally present in `text`, compared case-insensitively."""
    lowered = text.lower()
    return [token for token in FORBIDDEN if token.lower() in lowered]


def _walk_lane():
    """Walk the lane once, returning `(py_files, symlinks)`.

    `os.walk` rather than `Path.rglob` because rglob cannot REPORT what it declines to look at:
    it silently skips a directory reached via a symlink. Here the pruning is explicit, so a
    symlink is observed and handed back to the caller to refuse on.
    """
    py_files = []
    symlinks = []
    for dirpath, dirnames, filenames in os.walk(_THIS_DIR, followlinks=False):
        here = Path(dirpath)
        rel_parts = here.relative_to(_THIS_DIR).parts
        kept = []
        for name in sorted(dirnames):
            child = here / name
            if _dir_is_skipped(rel_parts + (name,)):
                continue
            if child.is_symlink():
                symlinks.append(child)
                continue
            kept.append(name)
        dirnames[:] = kept
        for name in sorted(filenames):
            if not name.endswith(".py"):
                continue
            path = here / name
            if path.is_symlink():
                symlinks.append(path)
                continue
            if path.resolve() == _THIS_FILE:
                continue
            py_files.append(path)
    return sorted(py_files), sorted(symlinks)


def _lane_py_files():
    """Every `.py` file under the lane directory, at ANY depth, except this fence's own file —
    which necessarily spells the `FORBIDDEN` tokens out as literal strings to define them (and
    would otherwise trip on itself) — and except the directories named by `ALWAYS_SKIP` (any
    depth) / `TOP_LEVEL_ONLY_SKIP` (lane root only). No filename exclusion exists: a curated
    allowlist here would just be the same defect the walk replaces.

    The walk is recursive because a `*.py` glob is blind to subpackages, and the next increment
    D68 anticipates (live order placement) is exactly the shape that lands in one.

    REFUSES (raises `FenceCannotVouchError`) if a symlinked directory or symlinked `.py` file is
    found under the lane: the walk cannot vouch for what is behind it, and staying green while
    blind is the failure mode this whole file exists to prevent.
    """
    py_files, symlinks = _walk_lane()
    if symlinks:
        named = ", ".join(str(p.relative_to(_THIS_DIR)) for p in symlinks)
        raise FenceCannotVouchError(
            f"the lane contains symlink(s) the credential fence cannot see behind: {named} — "
            "refusing to report an all-clear. Replace the symlink with a real file/directory "
            "inside the lane (or move its target under the lane) so the walk can scan it.")
    return py_files


def test_no_py_file_in_the_lane_references_any_credential_or_order_transmission_symbol():
    files = _lane_py_files()
    assert len(files) >= 5, (
        "the walk found suspiciously few files — sanity-check the fence itself before trusting "
        "an all-clear from it")
    for path in files:
        hits = _forbidden_hits(path.read_text())
        assert not hits, f"{path.relative_to(_THIS_DIR)} references {hits} — refusing"


def test_no_file_here_imports_kraken_client_private_machinery():
    for path in _lane_py_files():
        assert "kraken_client" not in path.read_text().lower(), (
            f"{path.relative_to(_THIS_DIR)} references kraken_client.py")


def test_the_fence_excludes_only_its_own_file_never_a_curated_list():
    """Anti-regression for the fence's own scope: proves the only lane source file left unscanned
    is this one, not a list someone could quietly grow back into an allowlist."""
    scanned = {p.resolve() for p in _lane_py_files()}
    top_level = {p.resolve() for p in _THIS_DIR.glob("*.py")}
    assert top_level - scanned == {_THIS_FILE}


def test_the_walk_skips_tooling_caches_and_vendored_envs():
    """Non-vacuity for `ALWAYS_SKIP`: the recursion must not start scanning interpreter caches or
    a vendored environment that happens to be dropped inside the lane."""
    junk_dir = _THIS_DIR / "__pycache__" / "_scope3_junk_scratch"
    junk_dir.mkdir(parents=True, exist_ok=True)
    junk = junk_dir / "vendored_scratch.py"
    try:
        junk.write_text('value = getenv("SOME_SETTING")\n')
        assert junk.resolve() not in {p.resolve() for p in _lane_py_files()}, (
            "the walk descended into a tooling cache — it will eventually trip on vendored code")
    finally:
        shutil.rmtree(junk_dir, ignore_errors=True)


def test_a_machine_generated_cache_is_skipped_at_ANY_depth_not_only_the_lane_root():
    """The `ALWAYS_SKIP` half of the two-set rule (TASK_910 cycle-3 BLOCKER, other direction).
    The fix for that BLOCKER must not be "just match `parts[0]`": a `__pycache__` sitting inside
    a lane subpackage is still interpreter output, and scanning it would make the fence's runtime
    and failure messages depend on whatever bytecode happens to be on disk.
    """
    subdir = _THIS_DIR / "_scope3_nested_cache_scratch"
    assert not subdir.exists()
    cache_dir = subdir / "__pycache__"
    cache_dir.mkdir(parents=True)
    try:
        junk = cache_dir / "vendored_scratch.py"
        junk.write_text('value = getenv("SOME_SETTING")\n')
        assert junk.resolve() not in {p.resolve() for p in _lane_py_files()}, (
            "the walk descended into a NESTED __pycache__ — ALWAYS_SKIP must hold at every depth")
    finally:
        shutil.rmtree(subdir, ignore_errors=True)


def test_an_ambiguous_dir_name_is_skipped_at_the_lane_root_only():
    """The `TOP_LEVEL_ONLY_SKIP` half: `<lane>/build/` is a packaging output tree and is skipped,
    which is the whole reason the name is in the skip list at all. Pinned so that the BLOCKER fix
    below cannot be "achieved" by simply deleting these names from the skip sets.
    """
    build_dir = _THIS_DIR / "build"
    assert not build_dir.exists(), (
        "a real `build/` exists in the lane root — this control cannot run without clobbering it")
    build_dir.mkdir()
    try:
        junk = build_dir / "vendored_scratch.py"
        junk.write_text('value = getenv("SOME_SETTING")\n')
        assert junk.resolve() not in {p.resolve() for p in _lane_py_files()}, (
            "a packaging output tree at the lane root was scanned — TOP_LEVEL_ONLY_SKIP is dead")
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)


def test_a_subdirectory_named_build_is_scanned_not_mistaken_for_a_vendored_env():
    """Non-vacuity for the TASK_910 cycle-3 BLOCKER, the direction that was actually broken.

    The shipped cycle-2 walk tested `parts[:-1]` — EVERY directory component at EVERY depth —
    against one flat skip set, so any subdirectory whose name collided with that set went
    invisible no matter how deep. Reproduced first-hand by the orchestrator: a
    `venue2/build/order_client.py` reading a credential env var and building an `add_order`
    frame left the whole fence at "7 passed, clean". `build/`/`dist/` are exactly what Python
    packaging produces — a full duplicate of every source file, credential-reading code included
    — and `env/` is an ordinary name for exchange-environment config, so the collision is not
    hypothetical. Plant the poison at `<lane>/<sub>/build/` and prove the fence goes RED.
    """
    subdir = _THIS_DIR / "_scope3_nested_build_scratch"
    assert not subdir.exists()
    nested = subdir / "build"
    nested.mkdir(parents=True)
    try:
        poison = nested / "order_client_scratch.py"
        poison.write_text('value = getenv("SOME_SETTING")\n')
        assert poison.resolve() in {p.resolve() for p in _lane_py_files()}, (
            "a .py file under a NESTED `build/` was not picked up — an ambiguous directory name "
            "is hiding lane source again, exactly as in TASK_910 cycle 3")
        with pytest.raises(AssertionError) as caught:
            test_no_py_file_in_the_lane_references_any_credential_or_order_transmission_symbol()
        assert str(poison.relative_to(_THIS_DIR)) in str(caught.value), (
            "the fence went RED, but not on the nested-`build/` file we planted — this control "
            "would otherwise pass for the wrong reason")
    finally:
        shutil.rmtree(subdir, ignore_errors=True)


def test_the_fence_refuses_rather_than_silently_skipping_a_symlinked_directory():
    """TASK_910 cycle-3 MAJOR. `Path.rglob`/`os.walk` do not descend into a directory reached via
    a symlink (verified in this environment), so `ln -s ../shared/venue_client.py venue/` — an
    ordinary refactor — would import and execute normally while never being scanned. Same failure
    class as the BLOCKER: the fence stays green over code it never read. The fix is fail-closed,
    so what this control asserts is a REFUSAL naming the symlink, not a clean scan.
    """
    subdir = _THIS_DIR / "_scope3_symlink_scratch"
    assert not subdir.exists()
    real = subdir / "real"
    real.mkdir(parents=True)
    try:
        (real / "venue_client_scratch.py").write_text('value = getenv("SOME_SETTING")\n')
        link = subdir / "linked"
        link.symlink_to("real", target_is_directory=True)
        assert (link / "venue_client_scratch.py").exists(), (
            "the symlink is not usable — this control would prove nothing")
        with pytest.raises(FenceCannotVouchError) as caught:
            _lane_py_files()
        assert str(link.relative_to(_THIS_DIR)) in str(caught.value), (
            "the fence refused, but without naming the symlink it cannot see behind")
    finally:
        shutil.rmtree(subdir, ignore_errors=True)


def test_the_fence_refuses_on_a_symlinked_py_file_too():
    """The file-level half of the same refusal. A symlinked `.py` file IS followed by the walk
    today, so this is not a silent-skip hole — but the fence still cannot vouch for it: its
    target lives outside what the walk enumerated, can change under it, and its failure message
    would name a path whose contents are somewhere else. Refusing keeps the guarantee legible.
    """
    subdir = _THIS_DIR / "_scope3_symlink_file_scratch"
    assert not subdir.exists()
    subdir.mkdir()
    try:
        target = subdir / "real_module_scratch.py"
        target.write_text("value = 1\n")
        link = subdir / "linked_module_scratch.py"
        link.symlink_to("real_module_scratch.py")
        with pytest.raises(FenceCannotVouchError) as caught:
            _lane_py_files()
        assert str(link.relative_to(_THIS_DIR)) in str(caught.value), (
            "the fence refused, but without naming the symlinked .py file")
    finally:
        shutil.rmtree(subdir, ignore_errors=True)


def test_a_new_file_not_yet_written_would_be_caught_without_touching_this_fence():
    """Non-vacuity / positive control for 'covers the NEXT increment': write a throwaway `.py`
    file into the lane directory carrying a forbidden token, confirm the fence goes RED on it,
    and remove it. Proves the mechanism, not just the current file list.

    The planted token is `getenv` — enough to trip the fence, and deliberately NOT a realistic
    credential name: if this process is killed between the write and the `finally`, the residue
    left in the tracked tree is an inert scratch file, not something that reads like a secret.
    """
    poison = _THIS_DIR / "_scope3_nonexistent_yet_scratch.py"
    assert not poison.exists()
    poison.write_text('value = getenv("SOME_SETTING")\n')
    try:
        assert poison.resolve() in {p.resolve() for p in _lane_py_files()}, (
            "a brand-new .py file in the lane was not picked up by the walk — the fence would "
            "have missed the next increment exactly like the old filename allowlist did")
        with pytest.raises(AssertionError) as caught:
            test_no_py_file_in_the_lane_references_any_credential_or_order_transmission_symbol()
        assert poison.name in str(caught.value), (
            "the fence went RED, but not on the file we planted — this control would otherwise "
            "pass for the wrong reason")
    finally:
        poison.unlink()


def test_a_new_file_one_directory_down_is_caught_not_just_at_the_top_level():
    """Non-vacuity for the RECURSION specifically (TASK_910 cycle-2 BLOCKER). The shipped fence
    globbed `*.py`, so a file at `<lane>/venue/order_client.py` reading both credential env vars
    and building an `AddOrder` frame passed all four fence tests clean — and a subpackage is
    exactly where a phase-2 order-placement increment would land. The top-level positive control
    above could not see this, which is why the blind spot survived mutation testing: plant the
    poison ONE DIRECTORY DOWN and prove the scanning test goes RED.
    """
    subdir = _THIS_DIR / "_scope3_subdir_scratch"
    assert not subdir.exists()
    subdir.mkdir()
    try:
        poison = subdir / "order_client_scratch.py"
        poison.write_text('value = getenv("SOME_SETTING")\n')
        assert poison.resolve() in {p.resolve() for p in _lane_py_files()}, (
            "a .py file one directory down was not picked up — the walk is not recursive")
        with pytest.raises(AssertionError) as caught:
            test_no_py_file_in_the_lane_references_any_credential_or_order_transmission_symbol()
        assert str(poison.relative_to(_THIS_DIR)) in str(caught.value), (
            "the fence went RED, but not on the subdirectory file we planted — this control "
            "would otherwise pass for the wrong reason")
    finally:
        shutil.rmtree(subdir, ignore_errors=True)


def test_ordinary_non_obfuscated_credential_and_order_idioms_are_caught():
    """Non-vacuity for the widened `FORBIDDEN` (TASK_910 cycle-2 MAJOR (a)). These are ORDINARY
    spellings — the kind someone reaches for without any intent to evade — and every one of them
    had zero token overlap with the shipped `FORBIDDEN`. Deliberate obfuscation (string assembly,
    `exec`, runtime name binding) is NOT covered and is not claimed to be; see the docstring.
    """
    ordinary = (
        'from os import environ as _e',
        'os_mod = __import__("os")',
        'os_mod = importlib.import_module("os")',
        'env = getattr(os, "environ")',
        'frame = {"method": "addOrder"}',
        'frame = {"method": "ADD_ORDER"}',
        'frame = {"method": "Add_Order"}',
        'headers = {"API_KEY": token}',
    )
    for snippet in ordinary:
        assert _forbidden_hits(snippet), f"{snippet!r} slipped past FORBIDDEN"
