#!/usr/bin/env python3
"""TASK_903 cycle-2 — the PROOF for the BLOCKER fix in ``test_903_constant_provenance.py``:
the 837-C1 gate can no longer destroy uncommitted work in the tree it runs from.

WHAT WAS WRONG (cycle 1). ``gate_837c1_fixture_smoke_drift_is_named`` ran ``walkforward.py
smoke`` in the LIVE working tree and then "self-cleaned" by computing
``git diff --name-only -- experiments/837_walkforward`` *after* the subprocess and passing
whatever came back to ``git checkout --``. A blanket after-the-fact sweep cannot tell "dirtied
by this run" from "was already dirty for an unrelated reason", so an unrelated uncommitted edit
in that directory was silently destroyed at exit 0 — and the ``finally`` block ran the revert
even when the smoke subprocess itself had failed. This is the run-34 incident class (a stray
``git stash`` stripping a concurrent lane's work), committed into a test every future run would
execute.

WHAT THIS FILE PROVES, by running the reconstructed old mechanism (see the header above
``_cycle1_destructive_selfclean`` for exactly how faithful that reconstruction is, and is not)
and the REAL new gate against a planted decoy:
  A1  OLD mechanism, successful smoke  -> the decoy is DESTROYED, silently.
  A2  OLD mechanism, FAILED smoke      -> the decoy is DESTROYED anyway (the `finally` fires on
                                          every exit path, including the ones that wrote
                                          nothing at all).
  B   NEW gate, dirty 837 directory    -> the gate SKIPS fail-closed; the decoy SURVIVES
                                          byte-identical and no other file is modified.
  C   NEW gate, clean 837 directory    -> the gate RUNS to completion, the smoke's 8 file
                                          rewrites all land in a throwaway worktree, the host
                                          tree ends with ZERO modifications under
                                          experiments/837_walkforward/, and an unrelated
                                          uncommitted edit elsewhere SURVIVES.

SAFETY OF THIS DEMONSTRATION ITSELF. No decoy is ever planted in the live tree and the live
tree is never written to. Every scenario runs inside a disposable detached ``git worktree``
(the "host tree" below) created in a temp directory; scenario C's new gate then creates a
second, nested throwaway worktree of its own. The only git commands issued against the live
repo are ``worktree add`` / ``worktree remove --force`` (both path-scoped to the throwaway
directory) and read-only ``status``. ``git worktree prune`` is deliberately NEVER run — it
takes no path filter and would delete the administrative record of any sibling worktree whose
directory is momentarily absent, live and unlocked ones included (cycle-3 MINOR 3). On SIGKILL
the live tree is byte-for-byte unchanged; a temp directory and a ``.git/worktrees/``
bookkeeping entry may leak — harmless, and a human can prune when no other lane is running.
The demo refuses to start if the live tree is dirty under ``experiments/837_walkforward/``.

Run: trader_env/bin/python experiments/892_action_ev/test_903_worktree_isolation_demo.py
Cost: two fixture smoke runs (~25s each) plus two worktree checkouts (23MB tracked) — ~1-2 min,
CPU only, fixture db only, never trading_data.db.
"""
import os
import shutil
import subprocess
import sys
import tempfile

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(EXP_DIR))
WF_REL = os.path.join("experiments", "837_walkforward")
DECOY_REL = os.path.join(WF_REL, "SPEC.md")               # a file walkforward.py NEVER writes
DECOY_OUTSIDE_REL = os.path.join("experiments", "892_action_ev", "SPEC.md")
DECOY_MARK = "\n<!-- TASK_903 DEMO DECOY: a concurrent lane's uncommitted work. -->\n"

if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)
import test_903_constant_provenance as prov                                       # noqa: E402


# =============================================================================
# Disposable host trees (never the live tree).
# =============================================================================
def _add_worktree():
    """Detached worktree of the live HEAD in a temp dir, with the gitignored fixture db
    symlinked in (read-only use: walkforward.py:220 opens every handle `?mode=ro`)."""
    wt = tempfile.mkdtemp(prefix="task903_demo_host_")
    os.rmdir(wt)
    add = subprocess.run(["git", "worktree", "add", "--detach", wt, "HEAD"],
                         cwd=REPO, capture_output=True, text=True)
    if add.returncode != 0:
        raise RuntimeError(f"git worktree add failed: {add.stderr[-400:]}")
    os.makedirs(os.path.join(wt, "fixtures"), exist_ok=True)
    os.symlink(os.path.join(REPO, "fixtures", "trading_fixture.db"),
               os.path.join(wt, "fixtures", "trading_fixture.db"))
    return wt


def _remove_worktree(wt):
    """Path-scoped teardown only. NO `git worktree prune` fallback (cycle-3 MINOR 3): prune is
    repo-wide, and it deletes the record — index, HEAD, reflog — of ANY worktree whose directory
    it cannot see at that instant, including another lane's live, unlocked one. A stale
    bookkeeping entry is harmless by comparison."""
    rm = subprocess.run(["git", "worktree", "remove", "--force", wt],
                        cwd=REPO, capture_output=True, text=True)
    if rm.returncode != 0:
        shutil.rmtree(wt, ignore_errors=True)
        print(f"    [warn] `git worktree remove --force {wt}` failed "
              f"({rm.stderr.strip()[-200:]}); directory deleted, stale .git/worktrees/ entry "
              f"may remain. Not pruning.")


def _plant(host, rel):
    """Append the decoy marker to a tracked file in the host tree; return its exact bytes."""
    path = os.path.join(host, rel)
    with open(path, "rb") as fh:
        original = fh.read()
    with open(path, "ab") as fh:
        fh.write(DECOY_MARK.encode())
    with open(path, "rb") as fh:
        return fh.read()


def _read(host, rel):
    with open(os.path.join(host, rel), "rb") as fh:
        return fh.read()


def _status(host, rel_scope):
    out = subprocess.run(["git", "status", "--porcelain", "--", rel_scope],
                         cwd=host, capture_output=True, text=True, check=True).stdout
    return [ln for ln in out.splitlines() if ln.strip()]


# =============================================================================
# The OLD mechanism. HOW MUCH TO TRUST THIS (cycle-3 NIT 7 — the earlier header claimed it was
# "reproduced verbatim from cycle 1", which is not checkable and is therefore withdrawn):
# test_903_constant_provenance.py was never committed at cycle 1 and was overwritten in place at
# cycle 2, so NO artifact of the cycle-1 source survives anywhere — nothing can be diffed
# against it, by anyone. What follows is a RECONSTRUCTION of the mechanism as the cycle-2
# review described it, vendored inline below so a reader can judge the comparison for
# themselves rather than take a provenance claim on faith. The three properties the
# demonstration actually turns on are all visible in the ten lines below and none of them are
# in dispute: (1) the smoke runs in the tree passed in, (2) the dirty set is computed AFTER the
# subprocess with a blanket `git diff --name-only -- experiments/837_walkforward`, and (3) that
# set is passed to `git checkout --` from a `finally`, so it fires on every exit path. Anything
# cosmetic about the original (names, logging, ordering) is NOT claimed to be reproduced.
# =============================================================================
def _cycle1_destructive_selfclean(host, smoke_args=("smoke",)):
    rc = None
    try:
        proc = subprocess.run(
            [sys.executable, os.path.join(host, WF_REL, "walkforward.py"), *smoke_args],
            cwd=host, capture_output=True, text=True, timeout=600)
        rc = proc.returncode
    finally:
        dirty = subprocess.run(["git", "diff", "--name-only", "--", WF_REL],
                               cwd=host, capture_output=True, text=True).stdout.split()
        if dirty:
            subprocess.run(["git", "checkout", "--"] + dirty, cwd=host, check=True)
    return rc


# =============================================================================
# Scenarios.
# =============================================================================
def scenario_a1_old_destroys(failures):
    host = _add_worktree()
    try:
        planted = _plant(host, DECOY_REL)
        rc = _cycle1_destructive_selfclean(host)
        after = _read(host, DECOY_REL)
        destroyed = after != planted
        print(f"    [{'PASS' if destroyed else 'FAIL'}] A1 (OLD mechanism, smoke rc={rc}): the "
              f"pre-existing uncommitted edit to {DECOY_REL} — a file walkforward.py never "
              f"writes — was {'DESTROYED, silently, with no warning' if destroyed else 'preserved'}"
              f". This is the reconstructed cycle-1 mechanism (see its header for the limits of "
              f"that reconstruction).")
        if not destroyed:
            failures.append("A1: the cycle-1 mechanism did NOT destroy the decoy — the "
                            "demonstration no longer reproduces the reported defect; re-derive "
                            "it before trusting the comparison below")
    finally:
        _remove_worktree(host)


def scenario_a2_old_destroys_even_when_smoke_fails(failures):
    host = _add_worktree()
    try:
        planted = _plant(host, DECOY_REL)
        # A subcommand argparse rejects: the subprocess exits non-zero having written NOTHING,
        # so every path the sweep finds is somebody else's work by construction.
        rc = _cycle1_destructive_selfclean(host, smoke_args=("smoke", "--no-such-flag-903"))
        after = _read(host, DECOY_REL)
        destroyed = after != planted
        verdict = ("DESTROYED anyway — the `finally` reverts on every exit path, so a run that "
                   "produced no side effects at all still eats a concurrent lane's work"
                   if destroyed else "preserved")
        print(f"    [{'PASS' if destroyed else 'FAIL'}] A2 (OLD mechanism, smoke FAILED "
              f"rc={rc}, wrote nothing): the decoy was {verdict}")
        if not destroyed:
            failures.append("A2: the cycle-1 mechanism did not destroy the decoy on a failed "
                            "smoke — re-derive the demonstration")
    finally:
        _remove_worktree(host)


def _run_new_gate_against(host):
    """Point the rebuilt gate at ``host`` instead of the live repo and run it. Returns the
    gate's own ``prov.Audit`` (findings / skipped / failures)."""
    saved = (prov.REPO, prov.WF_DIR)
    prov.REPO, prov.WF_DIR = host, os.path.join(host, WF_REL)
    try:
        audit = prov.Audit()
        prov.gate_837c1_fixture_smoke_drift_is_named(audit)
        return audit
    finally:
        prov.REPO, prov.WF_DIR = saved


def scenario_b_new_skips_fail_closed(failures):
    host = _add_worktree()
    try:
        planted = _plant(host, DECOY_REL)
        audit = _run_new_gate_against(host)
        after = _read(host, DECOY_REL)
        survived = after == planted
        modified = _status(host, WF_REL)
        only_decoy = [ln for ln in modified if not ln.endswith(DECOY_REL)]
        # The skip must be REGISTERED (counted + named + no finding), not just printed —
        # cycle-3 MAJOR 1: a skip that leaves no record lets the summary assert a
        # classification this run never made.
        ok = (survived and len(audit.skipped) == 1 and audit.skipped[0][0] == "837-C1"
              and not audit.findings and not audit.failures and not only_decoy)
        print(f"    [{'PASS' if ok else 'FAIL'}] B (NEW gate, 837 dirty): the gate refused "
              f"fail-closed and REGISTERED the refusal — skipped={[g for g, _ in audit.skipped]}, "
              f"findings={[g for g, _ in audit.findings]} (must be empty: a skipped gate "
              f"contributes no finding to the summary), failures={len(audit.failures)}; the "
              f"decoy {'SURVIVED byte-identical' if survived else 'was DESTROYED'} and no other "
              f"file under {WF_REL} was modified (host status: {modified})")
        if not ok:
            failures.append(f"B: survived={survived} skipped={audit.skipped} "
                            f"findings={audit.findings} gate_failures={audit.failures} "
                            f"other_modified={only_decoy}")
    finally:
        _remove_worktree(host)


def scenario_c_new_runs_and_dirties_nothing(failures):
    host = _add_worktree()
    try:
        # 837 stays CLEAN so the gate actually runs; the unrelated uncommitted work sits
        # elsewhere in the same tree, where the cycle-1 sweep would not have seen it but a
        # careless `git checkout`/`stash` widening would.
        planted_outside = _plant(host, DECOY_OUTSIDE_REL)
        audit = _run_new_gate_against(host)
        after_outside = _read(host, DECOY_OUTSIDE_REL)
        wf_modified = _status(host, WF_REL)
        outside_survived = after_outside == planted_outside
        ok = (not audit.failures and not audit.skipped and len(audit.findings) == 1
              and not wf_modified and outside_survived)
        print(f"    [{'PASS' if ok else 'FAIL'}] C (NEW gate, 837 clean): the gate RAN to "
              f"completion and recorded exactly one finding "
              f"({[g for g, _ in audit.findings]}, {len(audit.failures)} failures, "
              f"{len(audit.skipped)} skips); the host tree ends with {len(wf_modified)} "
              f"modification(s) under {WF_REL} — the smoke's 8 file rewrites all landed in its "
              f"own throwaway worktree — and the unrelated uncommitted edit at "
              f"{DECOY_OUTSIDE_REL} "
              f"{'SURVIVED byte-identical' if outside_survived else 'was DESTROYED'}")
        if not ok:
            failures.append(f"C: gate_failures={audit.failures} skipped={audit.skipped} "
                            f"findings={audit.findings} wf_modified={wf_modified} "
                            f"outside_survived={outside_survived}")
    finally:
        _remove_worktree(host)


def main():
    print("=" * 84)
    print("[903-demo] worktree isolation of the 837-C1 gate — OLD destroys / NEW preserves")
    print("=" * 84)

    try:
        live_dirty = prov.live_wf_dirty_paths()
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"[903-demo] REFUSING TO RUN (fail-closed): could not establish whether the live "
              f"tree is clean under {WF_REL} (`git status` failed: {exc}). Nothing was touched.")
        return 1
    if live_dirty:
        print(f"[903-demo] REFUSING TO RUN (fail-closed): the live tree is dirty under "
              f"{WF_REL} — {live_dirty}. This demo creates worktrees from HEAD and would not "
              f"be measuring the committed state. Nothing was touched.")
        return 1

    print("[903-demo] A — the cycle-1 mechanism, reconstructed (no cycle-1 artifact survives)")
    failures = []
    scenario_a1_old_destroys(failures)
    scenario_a2_old_destroys_even_when_smoke_fails(failures)

    print("[903-demo] B/C — the rebuilt gate")
    scenario_b_new_skips_fail_closed(failures)
    scenario_c_new_runs_and_dirties_nothing(failures)

    # Closing invariant: this demo must itself have left the live tree exactly as it found it.
    still_clean = prov.live_wf_dirty_paths()
    ok_live = not still_clean
    print(f"    [{'PASS' if ok_live else 'FAIL'}] the LIVE tree is still clean under {WF_REL} "
          f"after all four scenarios — every decoy was planted in a disposable worktree")
    if not ok_live:
        failures.append(f"the demo dirtied the LIVE tree under {WF_REL}: {still_clean}")

    print("=" * 84)
    if failures:
        print(f"[903-demo] FAIL — {len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[903-demo] PASS — the cycle-1 self-clean destroys a concurrent lane's uncommitted "
          "edit on BOTH a successful and a failed smoke; the rebuilt gate preserves it, "
          "refuses fail-closed on a dirty 837, and leaves zero modifications behind when it "
          "does run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
