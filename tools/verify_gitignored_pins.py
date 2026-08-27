#!/usr/bin/env python3
"""TASK_911 (D69) -- verify the identity of load-bearing gitignored artifacts.

``runs/`` is gitignored by design, so the clean-tree gate that every other ship ritual leans on
cannot see a file under it change at all. A backup restore or mtime-preserving sync can silently
overwrite a deployed repair with stale content -- measured 2026-08-25 (D69's forensics):
``runs/809_decision_rule_rescue/recompute_best.py`` regressed to its June revision on the
workstation, and the only in-repo detector was a red test the factory nearly deleted as
"probably stale". This module makes that class of regression loud instead of invisible.

Same doctrine as ``fixtures/fixture_identity.py`` (the fixture-DB pin) and
``mcts_trader/jax_agent/checkpoint_io.py``'s ``frame_symbols_sha256`` stamp (TASK_894): a missing
or corrupt pin REFUSES, never falls back to "no check happened". This module adds one more rule
those two didn't need, because their subject is a single file/DB each and always either fully
present or absent together: **a pinned file going MISSING is reported exactly as loudly as one
whose content diverges.** "Absent" is the failure mode the ``orch_wf875.sh`` row exists to catch
(see ``config/gitignored_pins.json``'s finding for that path) -- a verifier that quietly treats
"not found" as "nothing to check" is the same defect in a new costume.

Statuses, never conflated:
  OK       -- file present, sha256 matches the pin.
  DIVERGED -- the file's identity does NOT match the pin, or could not be established at all:
              a genuine content mismatch (the D69 class), a path that resolves OUTSIDE
              ``repo_root``, or a file present-but-unhashable (a directory, an unreadable
              file -- any OSError). All three are refusals, so they share one fail-closed
              status rather than each adding a branch every caller would have to learn; the
              ``message`` always names WHICH of the three it was.
  MISSING  -- file absent on this host. Always reported; never silently skipped.
  UNPINNED -- file present but the manifest records no known-good sha256 for it (``null``).
              Distinct from OK on purpose: presence alone is not verification.

``verify_pin`` NEVER raises: every one of those outcomes comes back as a structured result.
That is load-bearing, not tidiness -- the 897 suite calls this from ``pytest_sessionstart``,
where an escaping exception becomes a pytest INTERNALERROR and the legible refusal is lost at
exactly the moment it matters most.

``refuse_on_divergence`` raises on DIVERGED or MISSING. UNPINNED is a softer, named finding
(there is nothing to compare against yet) and does not raise on its own -- callers that want it
to be fatal too can check for it explicitly (see ``main()``'s exit-code contract below).

CLI:
    python -m tools.verify_gitignored_pins                       # verify the whole manifest
    python -m tools.verify_gitignored_pins --path runs/a --path runs/b   # verify a subset
    python -m tools.verify_gitignored_pins --promote-pin runs/foo/bar.py \\
        --provenance '...' --why '...' --evidence '...'          # deliberate re-pin ONLY

Exit codes (the gate-(iii) contract a future caller can rely on without re-deriving anything):
    0 -- every checked entry is OK.
    1 -- at least one checked entry is DIVERGED or MISSING (a refusal).
    2 -- no DIVERGED/MISSING, but at least one checked entry is UNPINNED (incomplete, not green).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = _REPO / "config" / "gitignored_pins.json"

_REQUIRED_ENTRY_FIELDS = ("path", "sha256", "provenance", "why_load_bearing")

# Fields whose VALUE must be a non-empty string, not merely a present key. ``sha256`` is
# deliberately NOT here: ``null`` is a documented, meaningful value for it (a load-bearing file
# for which no known-good hash was ever recorded -- ``runs/875_wf/orch_wf875.sh``), and such an
# entry always reports MISSING/UNPINNED, never OK.
#
# Checking presence alone was a laundering hole: ``--promote-pin`` with no ``--provenance``/
# ``--why`` wrote ``null`` into both audit fields, and the resulting entry then passed the
# required-field check on every subsequent read forever, because the KEYS existed. The promote
# path exists precisely to guarantee that audit trail, so it is enforced on BOTH sides -- refused
# on write (``promote_pin``) and caught on read (``load_manifest``), so an entry laundered by an
# older build of this tool cannot survive quietly either.
_REQUIRED_TRUTHY_ENTRY_FIELDS = ("path", "provenance", "why_load_bearing")

# --provenance / --why / --evidence: all three are the audit trail, all three are mandatory for a
# promotion. (flag, what it must name) -- used for both the refusal and the CLI help text.
_PROMOTE_REQUIRED_ARGS = (
    ("--provenance", "where this content came from"),
    ("--why", "why this file is load-bearing"),
    ("--evidence", "the re-derivation evidence"),
)


class PinVerificationError(RuntimeError):
    """The manifest itself could not be loaded, a promotion was attempted without its full audit
    trail, a manifest path escaped ``repo_root``, or a caller asked to be raised at on a
    DIVERGED/MISSING result (see ``refuse_on_divergence``)."""


def _resolve_within_repo(repo_root, rel):
    """Resolve a manifest ``path`` against ``repo_root`` and REFUSE if it lands outside it.

    ``Path(repo_root) / rel`` alone does no containment check at all: an entry reading
    ``"../elsewhere/secret.txt"`` (or an absolute path, or a symlink planted inside ``runs/``)
    resolves and gets hashed against a file in a completely different tree, and reports OK. The
    manifest is tracked and reviewed so this is not externally reachable -- but nothing asserted
    it, and a pin that verifies a file outside the tree it claims to pin is a false green.

    Containment is checked on ``os.path.realpath`` of BOTH sides. This repo has been bitten twice
    by realpath-vs-abspath divergence (TASK_886/888): a ``startswith`` on unnormalized strings
    catches a literal ``..`` but sails straight past a symlink escape. Same shape as TASK_905's
    ``_require_within_fold_root`` (experiments/837_walkforward/walkforward.py:1411).

    Raises PinVerificationError, and does so BEFORE the caller can open the path -- that ordering
    is the guarantee. ``verify_pin`` catches it and folds it into a structured DIVERGED result
    (it must never raise); ``promote_pin`` lets it propagate, like its other refusals.
    """
    root_r = os.path.realpath(repo_root)
    p_r = os.path.realpath(os.path.join(root_r, rel))
    if not (p_r == root_r or p_r.startswith(root_r + os.sep)):
        raise PinVerificationError(
            f"manifest path {rel!r} resolves to {p_r}, OUTSIDE repo root {root_r} -- refusing to "
            f"hash it. A pin must name a file inside the tree it pins; hashing anything else "
            f"would report a false OK about a file this manifest does not describe.")
    return Path(p_r)


def sha256_of(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def load_manifest(manifest_path=DEFAULT_MANIFEST):
    """Load + validate the tracked manifest. Raises PinVerificationError (never returns a
    partial/default manifest) if it is missing, unreadable, or any entry is missing a required
    field OR carries a null/blank VALUE in one of the audit fields -- no fail-open on a deleted
    oracle (the same rule fixture_identity.py follows), and no fail-open on an entry whose audit
    trail is present-but-empty (see ``_REQUIRED_TRUTHY_ENTRY_FIELDS`` for why that distinction is
    the whole point)."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise PinVerificationError(
            f"gitignored-pins manifest missing: {manifest_path} -- refusing (no fail-open on a "
            f"deleted oracle).")
    try:
        with open(manifest_path) as fh:
            manifest = json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        raise PinVerificationError(
            f"gitignored-pins manifest unreadable/corrupt: {manifest_path} ({e}) -- refusing."
        ) from e
    pins = manifest.get("pins")
    if not isinstance(pins, list):
        raise PinVerificationError(
            f"gitignored-pins manifest {manifest_path} has no 'pins' list -- refusing.")
    for i, entry in enumerate(pins):
        missing = [f for f in _REQUIRED_ENTRY_FIELDS if f not in entry]
        if missing:
            raise PinVerificationError(
                f"gitignored-pins manifest {manifest_path} pin[{i}] ({entry.get('path')!r}) is "
                f"missing field(s) {missing} -- refusing.")
        blank = [f for f in _REQUIRED_TRUTHY_ENTRY_FIELDS
                 if not (isinstance(entry[f], str) and entry[f].strip())]
        if blank:
            raise PinVerificationError(
                f"gitignored-pins manifest {manifest_path} pin[{i}] ({entry.get('path')!r}) has "
                f"null/blank/non-string value(s) for {blank} -- refusing. A pin whose "
                f"provenance or why_load_bearing is empty carries no audit trail, which is the "
                f"one thing this manifest exists to carry; the key being present is not enough. "
                f"Fix the entry by hand, or re-run --promote-pin with --provenance/--why.")
        sha = entry["sha256"]
        if sha is not None and not (isinstance(sha, str) and sha.strip()):
            raise PinVerificationError(
                f"gitignored-pins manifest {manifest_path} pin[{i}] ({entry.get('path')!r}) has "
                f"a blank/non-string sha256 ({sha!r}) -- refusing. sha256 must be a non-empty "
                f"string, or exactly null (meaning: no known-good hash was ever recorded for "
                f"this path, which reports MISSING/UNPINNED and never OK). An empty string is "
                f"neither: it is a broken pin.")
    return manifest


def verify_pin(entry, repo_root=_REPO):
    """Verify ONE manifest entry against the live file on disk. Never raises for an ordinary
    mismatch/absence -- returns a structured result dict so callers (a pytest test, a CLI, a
    future gate-(iii) wiring) can decide what to do with it themselves.

    Returns: {"path", "status" in {OK, DIVERGED, MISSING, UNPINNED}, "expected", "actual",
              "message"} -- "message" always names expected vs actual in plain text.
    """
    rel = entry["path"]
    expected = entry.get("sha256")
    # Containment FIRST -- before .exists(), before any read. An escaping entry is a manifest
    # defect and must be reported as that, not as a bland MISSING (if it happens not to exist)
    # or, worse, as OK (if it does).
    try:
        p = _resolve_within_repo(repo_root, rel)
    except PinVerificationError as e:
        return {
            "path": rel, "status": "DIVERGED", "expected": expected, "actual": None,
            "message": f"DIVERGED (path escape): {e}",
        }
    if not p.exists():
        return {
            "path": rel, "status": "MISSING", "expected": expected, "actual": None,
            "message": (
                f"MISSING: {rel} not found on this host "
                f"(expected sha256 {expected or '<none ever recorded>'})"),
        }
    try:
        actual = sha256_of(p)
    except OSError as e:
        # A pin pointing at a directory (IsADirectoryError) or an unreadable file
        # (PermissionError) must not blow this function up: verify_pin's whole contract is a
        # structured result, and its loudest caller is pytest_sessionstart, where a raise
        # becomes an INTERNALERROR instead of a legible pytest.exit. Unhashable == identity NOT
        # established == refusal, so it folds into DIVERGED, never into OK/UNPINNED.
        return {
            "path": rel, "status": "DIVERGED", "expected": expected, "actual": None,
            "message": (
                f"DIVERGED (unhashable): {rel} exists but could not be read "
                f"({type(e).__name__}: {e}) -- its identity could NOT be established, so it is "
                f"refused rather than trusted (expected sha256 "
                f"{expected or '<none ever recorded>'})"),
        }
    if expected is None:
        return {
            "path": rel, "status": "UNPINNED", "expected": None, "actual": actual,
            "message": (
                f"UNPINNED: {rel} is present (sha256 {actual}) but the manifest records no "
                f"known-good hash for it yet -- cannot verify, and will NOT be silently trusted. "
                f"Promote deliberately with --promote-pin if this content is the intended "
                f"baseline."),
        }
    if actual != expected:
        return {
            "path": rel, "status": "DIVERGED", "expected": expected, "actual": actual,
            "message": f"DIVERGED: {rel} expected sha256 {expected}, got {actual}",
        }
    return {
        "path": rel, "status": "OK", "expected": expected, "actual": actual,
        "message": f"OK: {rel} matches pinned sha256 {expected[:12]}...",
    }


def verify_all(manifest_path=DEFAULT_MANIFEST, repo_root=_REPO, paths=None):
    """Verify every pin in the manifest, or only the entries whose ``path`` is in ``paths`` if
    given. Returns a list of result dicts (see ``verify_pin``) in manifest order. Raises
    PinVerificationError only if the manifest itself can't be loaded or ``paths`` names
    something not in it -- individual file mismatches/absences are returned, never raised,
    letting the caller compose its own refusal policy (see ``refuse_on_divergence``)."""
    manifest = load_manifest(manifest_path)
    entries = manifest["pins"]
    if paths is not None:
        wanted = list(paths)
        by_path = {e["path"]: e for e in entries}
        unknown = [p for p in wanted if p not in by_path]
        if unknown:
            raise PinVerificationError(
                f"verify_all() asked for path(s) not present in {manifest_path}: {unknown}")
        entries = [by_path[p] for p in wanted]
    return [verify_pin(e, repo_root) for e in entries]


def refuse_on_divergence(results, label="[gitignored-pins]"):
    """Raise PinVerificationError if ANY result is DIVERGED or MISSING. UNPINNED/OK results
    never raise here -- UNPINNED is a named, non-fatal finding the caller can still choose to
    escalate itself (see main()'s exit-2 contract)."""
    bad = [r for r in results if r["status"] in ("DIVERGED", "MISSING")]
    if bad:
        lines = "\n".join(f"  {r['message']}" for r in bad)
        raise PinVerificationError(
            f"{label} REFUSED -- {len(bad)} pinned gitignored file(s) diverged or went "
            f"missing:\n{lines}")


# --------------------------------------------------------------------------------------------
# Promote path (scope item 3) -- deliberate-only re-pin, mirrors fixtures/make_fixture.py's
# --promote-fixture-identity discipline: never automatic, never silent, always names the
# re-derivation evidence in the entry it writes so that evidence ships in the commit.
# --------------------------------------------------------------------------------------------
def promote_pin(path, provenance, why_load_bearing, evidence,
                 manifest_path=DEFAULT_MANIFEST, repo_root=_REPO):
    """Seed or update ONE manifest entry from the file's ACTUAL current sha256. Requires ALL
    THREE audit arguments -- ``provenance``, ``why_load_bearing`` and ``evidence`` -- to be
    non-empty, and refuses the whole promotion if any is missing or blank, so a promotion can
    never be a silent re-pin over an unexplained divergence (the D69 rule: 'never quietly bake a
    divergence in as the new truth').

    All three, not just ``evidence``: they were optional argparse flags defaulting to None, so
    ``--promote-pin X --evidence '...'`` exited 0, printed RE-PINNED, and wrote ``"provenance":
    null, "why_load_bearing": null`` -- a re-pin with no audit trail, which then read back
    clean forever. The refusal is checked BEFORE the file is hashed and before the manifest is
    opened for writing, so a refused promotion leaves the manifest byte-identical."""
    for flag, what in _PROMOTE_REQUIRED_ARGS:
        value = {"--provenance": provenance, "--why": why_load_bearing,
                 "--evidence": evidence}[flag]
        if not value or not str(value).strip():
            raise PinVerificationError(
                f"--promote-pin refuses without {flag}: it must name {what}, so the audit trail "
                f"ships in the commit -- never a silent re-pin. A promotion without one is "
                f"exactly the laundered re-pin this path exists to prevent.")
    p = _resolve_within_repo(repo_root, path)
    if not p.exists():
        raise PinVerificationError(f"--promote-pin: {path} does not exist on this host -- "
                                    f"nothing to hash.")
    try:
        new_sha = sha256_of(p)
    except OSError as e:
        raise PinVerificationError(
            f"--promote-pin: {path} exists but could not be read ({type(e).__name__}: {e}) -- "
            f"refusing to write a pin whose hash could not be computed.") from e
    manifest = load_manifest(manifest_path)
    entries = manifest["pins"]
    existing = next((e for e in entries if e["path"] == path), None)
    if existing is not None:
        old_sha = existing.get("sha256")
        existing["sha256"] = new_sha
        existing["provenance"] = provenance
        existing["why_load_bearing"] = why_load_bearing
        existing["promotion_evidence"] = evidence
        action = f"RE-PINNED (was {old_sha!r})"
    else:
        entries.append({
            "path": path, "sha256": new_sha, "provenance": provenance,
            "why_load_bearing": why_load_bearing, "promotion_evidence": evidence,
        })
        action = "NEW PIN"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    return f"{action}: {path} -> sha256 {new_sha}"


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--repo-root", default=str(_REPO),
                     help="root that manifest paths are resolved relative to (default: this "
                          "repo). Mainly for tests exercising the CLI against a scratch tree.")
    ap.add_argument("--path", action="append", default=None,
                     help="restrict the check to this manifest path; repeatable")
    ap.add_argument("--promote-pin", metavar="PATH", default=None,
                     help="deliberate re-pin: seed/update PATH's entry from its actual current "
                          "sha256. REQUIRES all three of --provenance/--why/--evidence, and "
                          "refuses (exit 1, manifest untouched) if any is missing or blank.")
    ap.add_argument("--provenance", default=None,
                     help="where this content came from -- required by --promote-pin")
    ap.add_argument("--why", default=None,
                     help="why this file is load-bearing -- required by --promote-pin")
    ap.add_argument("--evidence", default=None,
                     help="re-derivation evidence, named -- required by --promote-pin")
    args = ap.parse_args(argv)

    if args.promote_pin is not None:
        try:
            msg = promote_pin(args.promote_pin, args.provenance, args.why, args.evidence,
                               manifest_path=args.manifest, repo_root=args.repo_root)
        except PinVerificationError as e:
            print(str(e), file=sys.stderr)
            return 1
        print(msg)
        return 0

    try:
        results = verify_all(manifest_path=args.manifest, repo_root=args.repo_root,
                              paths=args.path)
    except PinVerificationError as e:
        print(str(e), file=sys.stderr)
        return 1

    for r in results:
        print(f"[{r['status']}] {r['message']}")

    bad = [r for r in results if r["status"] in ("DIVERGED", "MISSING")]
    unpinned = [r for r in results if r["status"] == "UNPINNED"]
    if bad:
        print(f"\nREFUSED -- {len(bad)} entr{'y' if len(bad) == 1 else 'ies'} DIVERGED or "
              f"MISSING.", file=sys.stderr)
        return 1
    if unpinned:
        print(f"\nINCOMPLETE -- {len(unpinned)} entr{'y' if len(unpinned) == 1 else 'ies'} "
              f"UNPINNED (no known-good hash recorded); nothing diverged, but this run makes "
              f"no claim about {'it' if len(unpinned) == 1 else 'them'}.", file=sys.stderr)
        return 2
    print(f"\nOK -- all {len(results)} checked entries match their pin.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
