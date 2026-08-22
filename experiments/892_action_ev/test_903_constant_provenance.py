#!/usr/bin/env python3
"""TASK_903 — the constant-provenance audit, pointed at 892's calibration table and 875's
walk-forward numbers (run-39's transferable question, asked of two lanes that had never been
asked it: *is each shipped constant the exact output of the command its own comment
advertises?*).

This is the gate-(i) ENUMERATION as CODE: ``MANIFEST`` below lists every shipped constant in
the 892 and 875/837 lanes whose comment or artifact text advertises a generating command,
seed, or reproduction claim. "We asked" is checkable by running this file — it is not an
assertion in a report. Standalone script (892 house style: an ``Audit`` collector, PASS/FAIL
summary), not pytest. Run:
    trader_env/bin/python experiments/892_action_ev/test_903_constant_provenance.py

EXIT-CODE CONTRACT (cycle-3 MAJOR-1 fix — a skip used to exit 0):
    0  every enumerated gate RAN and every check passed.
    1  at least one check FAILED — a shipped constant no longer matches its advertised
       command, or a finding this file pinned has itself moved.
    2  no check failed, but at least one gate was SKIPPED fail-closed. The audit is
       INCOMPLETE: the skipped gates' classifications were NOT established by this run, are
       NOT claimed in the summary, and must not be quoted from it. A SKIP IS NEVER EXIT 0.

SUMMARY CONSTRUCTION (same fix): the closing summary is built ONLY from ``Audit.findings`` —
entries a gate appends *as it concludes*. There is no hardcoded sentence listing the
classifications; a gate that skipped, failed, or crashed contributes nothing, so this file
cannot certify a provenance finding it did not make on this run. Every declared gate must
reach one of {finding, skip, failure}; a gate that reaches none is reported as a failure
rather than silently vanishing from the report.

SIDE-EFFECT DISCIPLINE (cycle-2 BLOCKER fix, see the 837-C1 block comment below): this file
NEVER writes to, and never reverts anything in, the working tree it is run from. The one gate
that needs to execute a side-effecting command (``walkforward.py smoke``, which rewrites eight
tracked files) runs it inside a throwaway detached ``git worktree`` — created outside the repo,
checked — and refuses fail-closed if ``experiments/837_walkforward/`` is dirty to begin with.
No ``git checkout``/``stash``/``reset``/``add``/``worktree prune`` is issued anywhere in this
module. The isolation is demonstrated, not asserted, by ``test_903_worktree_isolation_demo.py``
beside this file.

Full narrative (why each classification, the sensitivity read for every drift found, the
DEFERRED-EXPENSIVE cost estimates) is in ``PROVENANCE_AUDIT_903.md`` beside this file — this
module is the checkable skeleton that document reports numbers from; it does not re-derive the
DEFERRED-EXPENSIVE items (that would violate the cost-discipline invariant this row was warned
about explicitly) and it does not touch ``experiments/897_corrected_809/`` (a concurrent lane's
live, uncommitted work) or re-run either program's real leg (out of scope, D47/875 stand).

Classifications used (TASK_903 scope item 3):
  REPRODUCES                  — re-run (or a static comparison against an already-shipped
                                 companion artifact) matches the shipped constant exactly.
  DRIFTS-with-named-cause     — re-run differs from the shipped constant, and the cause is
                                 identified and dated. Sensitivity noted per gate (iv).
  PROVENANCE-UNRECOVERABLE    — the advertised claim omits information (a seed, a script, an
                                 access path) needed to reproduce it at all, from this session.
  DEFERRED-EXPENSIVE          — the advertised command is real and precise but too costly to
                                 run inside this row (measured, not guessed — see the doc).
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(EXP_DIR))
WF_DIR = os.path.join(REPO, "experiments", "837_walkforward")
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)

import disclosures as disclosures_mod                          # noqa: E402
import calibration as calibration_mod                          # noqa: E402
import report as report_mod                                    # noqa: E402


EXIT_OK = 0
EXIT_FAILED = 1
EXIT_INCOMPLETE = 2

EXIT_CODE_CONTRACT = (
    "EXIT CODES — 0: every enumerated gate ran and every check passed. "
    "1: at least one check FAILED. "
    "2: nothing failed, but >=1 gate was SKIPPED fail-closed, so this run is INCOMPLETE and "
    "makes NO provenance claim about the skipped gates. A skip never exits 0."
)


class Audit:
    """Per-gate record of what THIS RUN actually established.

    ``findings`` is the only source the closing summary is built from, which is the whole
    point (cycle-3 MAJOR 1): a gate appends to it *as it concludes*, so a gate that skipped,
    failed, or crashed contributes no sentence and the audit cannot certify a classification
    it never made. ``skipped`` and ``failures`` are named and counted separately, and both
    move the process off exit 0.
    """

    def __init__(self):
        self.findings = []      # list[(gate_id, classification established by this run)]
        self.skipped = []       # list[(gate_id, why it could not be established)]
        self.failures = []      # list[(gate_id, what did not hold)]

    def record(self, gate_id, classification):
        """A gate that RAN to a conclusion registers its classification here."""
        self.findings.append((gate_id, classification))

    def skip(self, gate_id, reason):
        print(f"    [SKIP] {gate_id}: {reason}")
        self.skipped.append((gate_id, reason))

    def fail(self, gate_id, detail):
        self.failures.append((gate_id, detail))

    def accounted(self):
        return {g for g, _ in self.findings} | {g for g, _ in self.skipped} \
            | {g for g, _ in self.failures}


# =============================================================================
# GATE (i) — THE ENUMERATION. Every row is a constant TASK_903 asked about. This list IS the
# checkable "we asked" — a new shipped constant advertising a command must be added here before
# this file can be said to cover the lane again.
# =============================================================================
MANIFEST = [
    {
        "id": "892-C1", "file": "experiments/892_action_ev/disclosures.py:338-363",
        "constant": "CALIBRATION_TABLE",
        "advertised_command": (
            "python calibration.py --n-trials-calib 200 --n-trials-valid 200 --n-boot 400 "
            "--seed 31337 --calibration-margin 0.03 --block-counts 2 3 4 5 6 10 20 80"
        ),
        "class": "cheap-exact",
    },
    {
        "id": "892-C2", "file": "experiments/892_action_ev/disclosures.py:238-245",
        "constant": "NULL_COVERAGE_TABLE / NULL_COVERAGE_TABLE_OVERLAPPING",
        "advertised_command": (
            "report.block_bootstrap_delta_ci on two independent samples; block=21, n_boot=2000, "
            "1000 trials per row, 'the data seed and the bootstrap seed both varied per trial' "
            "(disclosures.py:230-237 — NO concrete seed value or script is given)"
        ),
        "class": "unrecoverable",
    },
    {
        "id": "892-C3", "file": "experiments/892_action_ev/disclosures.py:311-330",
        "constant": "SPEC.md body sha256 prefix '8408970d...' (Amendment 1 provenance stamp)",
        "advertised_command": "sha256sum experiments/892_action_ev/SPEC.md",
        "class": "cheap-exact",
    },
    {
        "id": "892-C4",
        "file": "runs/892_action_ev/calibration_real_leg.json "
               "(experiments/892_action_ev/manifest_892_p1_policy.txt:29)",
        "constant": "the higher-precision CALIBRATION_TABLE CALIBRATION_PRECISION_DISCLOSURE "
                   "says is 'REQUIRED before the real leg launches'",
        "advertised_command": (
            "trader_env/bin/python experiments/892_action_ev/calibration.py "
            "--n-trials-calib 2000 --n-trials-valid 2000 --n-boot 2000 "
            "--out runs/892_action_ev/calibration_real_leg.json"
        ),
        "class": "deferred-expensive",
    },
    {
        "id": "892-C5", "file": "experiments/892_action_ev/disclosures.py:324-326",
        "constant": "'independent cross-check across three dependence models' "
                   "(k*~3.25 at 2 blocks, ~2.0 at 3-5, ~1.5 at 30)",
        "advertised_command": "none in tree — no script, no seed, approximate ('~') figures only",
        "class": "unrecoverable",
    },
    {
        # CYCLE-2 ADDITION. Cycle 1's MANIFEST claimed to be THE complete checkable enumeration
        # and missed this one: ZERO_WIDTH_FLOOR's comment makes an explicit, first-person
        # reproduction claim ("Measured directly (reproduced here and cross-checked against an
        # independent probe...)" + "MEASURED MARGIN: ...") — exactly the shape TASK_903 scope
        # item 1 says to enumerate. It already had EXECUTABLE coverage in the lane
        # (test_action_ev_892.py:1474 gate_zero_width_ci_never_bands, at the 1e-9/1e-6 jitter
        # scales, n_blocks 33/35) — the defect was the enumeration's completeness, not the
        # constant's. Re-asked here anyway, because gate (i) exists so that "we asked" is
        # itself checkable, and asking turned up a (non-material) drift — see the gate below.
        "id": "892-C6", "file": "experiments/892_action_ev/report.py:214-251",
        "constant": "ZERO_WIDTH_FLOOR = 1e-4 (and the MEASURED MARGIN claims in its comment)",
        "advertised_command": (
            "in-comment reproduction claim, no CLI: 'Measured directly (reproduced here and "
            "cross-checked against an independent probe from the factory orchestrator): "
            "holding the two sides' relative shape fixed and varying ONLY the overall scale "
            "across four decades (per-block jitter std 1e-9, 1e-6, 1e-4, and an O(1) healthy "
            "case), ci_width / jitter_std is CONSTANT to within Monte-Carlo noise' + 'MEASURED "
            "MARGIN: at 1e-9-scale per-block jitter the emitted width is O(1e-9 to 1e-10) ... "
            "at 1e-6-scale jitter it is O(1e-7)'. Existing executable coverage: "
            "test_action_ev_892.py:1474 gate_zero_width_ci_never_bands"
        ),
        "class": "cheap-exact",
    },
    {
        "id": "837-C1", "file": "experiments/837_walkforward/results_fixture_smoke.json",
        "constant": "the full fixture-smoke document (folds, corridors, negative-test messages, "
                   "aggregate)",
        "advertised_command": "trader_env/bin/python experiments/837_walkforward/walkforward.py smoke",
        "class": "cheap-exact",
    },
    {
        "id": "875-C1",
        "file": "experiments/LEDGER.md row '875/wf-generalization' -> "
               "gpu-server:runs/875_wf/f{1..5}_s{0,1,2}/results_wf875_*.json + grade_correct.py "
               "+ orch_wf875.sh",
        "constant": "every per-fold per-seed net@21 and teacher-floor number quoted in the "
                   "875/wf-generalization LEDGER row (the numbers underpinning 'no OOS edge')",
        "advertised_command": (
            "the orch_wf875.sh battery, gpu-server GPUs 1-3, tmux 'wf875', 8h57m wall "
            "(LEDGER:97) -- OR, cheaper: rsync gpu-server:~/Trader/runs/875_wf/ to the "
            "workstation so the existing artifacts can be inspected without re-running anything"
        ),
        "class": "unreachable-from-workstation",
    },
]

# Every gate this file promises to run. main() checks that each one reached a conclusion
# (finding / skip / failure); one that reached none crashed, and that is itself a failure —
# it must never just disappear from the summary.
DECLARED_GATES = ["gate-(i)", "892-C1", "892-C3", "892-C4", "892-C2", "892-C5", "892-C6",
                  "837-C1", "875-C1"]


def gate_enumeration_complete(audit):
    gid = "gate-(i)"
    ids = [m["id"] for m in MANIFEST]
    ok = len(ids) == len(set(ids)) and len(MANIFEST) >= 8
    print(f"    [{'PASS' if ok else 'FAIL'}] gate (i): MANIFEST enumerates {len(MANIFEST)} "
         f"distinct shipped constants across the 892 and 875/837 lanes, each carrying a file "
         f"locus and its advertised command/claim verbatim")
    if not ok:
        audit.fail(gid, "MANIFEST malformed or too short to be a real enumeration")
        return
    audit.record(gid, f"the enumeration is checkable and lists {len(MANIFEST)} distinct "
                      f"constants ({', '.join(ids)})")


# =============================================================================
# 892-C1 — CALIBRATION_TABLE. Cheap, exact, re-run through the real module (no re-implementation).
# =============================================================================
def gate_892c1_calibration_table_reproduces(audit):
    gid = "892-C1"
    table = calibration_mod.build_calibration_table(
        [2, 3, 4, 5, 6, 10, 20, 80], n_trials_calib=200, n_trials_valid=200, n_boot=400,
        seed=31337, calibration_margin=0.03)
    shipped = disclosures_mod.CALIBRATION_TABLE
    mismatches = []
    for nb, entry in shipped.items():
        got = table.get(nb)
        if got is None:
            mismatches.append(f"block count {nb} missing from re-run")
            continue
        for field in ("k", "achieved", "validation_coverage_overlapping",
                      "validation_coverage_iid", "validation_n_overlapping",
                      "validation_n_iid"):
            if got.get(field) != entry.get(field):
                mismatches.append(f"{nb}.{field}: shipped={entry.get(field)!r} "
                                  f"re-run={got.get(field)!r}")
    ok = not mismatches
    print(f"    [{'PASS' if ok else 'FAIL'}] 892-C1: disclosures.CALIBRATION_TABLE "
         f"({len(shipped)} rows) is BIT-IDENTICAL to a fresh run of its advertised command "
         f"('892-C1' in TASK_903's manifest) — REPRODUCES")
    if not ok:
        audit.fail(gid, f"DRIFTS: {mismatches}")
        return
    audit.record(gid, f"REPRODUCES — disclosures.CALIBRATION_TABLE is bit-identical to a fresh "
                      f"run of its advertised command, every field, all {len(shipped)} rows")


# =============================================================================
# 892-C3 — the SPEC.md body sha256 provenance stamp.
# =============================================================================
def gate_892c3_spec_hash_reproduces(audit):
    gid = "892-C3"
    spec_path = os.path.join(EXP_DIR, "SPEC.md")
    try:
        with open(spec_path, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
    except OSError as exc:
        audit.fail(gid, f"the enumerated artifact {spec_path} could not be read: {exc}")
        print(f"    [FAIL] 892-C3: SPEC.md unreadable: {exc}")
        return
    advertised_prefix = "8408970d"
    ok = digest.startswith(advertised_prefix)
    print(f"    [{'PASS' if ok else 'FAIL'}] 892-C3: sha256sum of SPEC.md as it stands is "
         f"{digest[:12]}... — {'matches' if ok else 'does NOT match'} the advertised prefix "
         f"'{advertised_prefix}...' (disclosures.py's Amendment-1 provenance stamp) — REPRODUCES")
    if not ok:
        audit.fail(gid, f"DRIFTS: SPEC.md sha256 is {digest}, advertised {advertised_prefix}...")
        return
    audit.record(gid, f"REPRODUCES — sha256sum of SPEC.md is {digest[:16]}..., matching the "
                      f"advertised '{advertised_prefix}...' provenance stamp")


# =============================================================================
# 892-C4 — the higher-precision recalibration disclosures.py's own CALIBRATION_PRECISION_
# DISCLOSURE says is REQUIRED before the real leg launches. DEFERRED-EXPENSIVE to RE-RUN (the
# --n-trials-calib 2000 command was measured, not guessed: ONE block count alone did not finish
# in 128s of wall time under these settings on this workstation -- default block-count list is
# 17, so the full advertised command is on the order of tens of minutes to a few hours of
# single-core CPU time; see PROVENANCE_AUDIT_903.md). What IS checked here, cheaply, is a STATIC
# fact requiring no Monte Carlo: does the constant this precondition was about
# (disclosures.CALIBRATION_TABLE, embedded in every shipped 892 report) actually equal the
# higher-precision file that was in fact produced and sits, unconsumed, beside the real-leg
# reports? Answering this needs zero new compute -- both files already exist.
#
# CYCLE-3 MAJOR-2 FIX: runs/892_action_ev/ is GITIGNORED, so on a fresh clone, on CI, on the
# server, and in any git worktree, this input is simply absent. The skip below used to be a
# bare `return` -- registered nowhere, counted nowhere -- while the closing banner went on
# asserting this gate's DRIFT finding anyway. It is now a first-class, named, counted skip,
# which also removes the finding from the summary and moves the process off exit 0.
# =============================================================================
def gate_892c4_real_leg_calibration_not_wired_in(audit):
    gid = "892-C4"
    real_leg_path = os.path.join(REPO, "runs", "892_action_ev", "calibration_real_leg.json")
    if not os.path.exists(real_leg_path):
        audit.skip(gid, f"{os.path.relpath(real_leg_path, REPO)} is absent from this checkout "
                        f"(gitignored run artifact — absent on any fresh clone, on CI, on the "
                        f"server, and in every git worktree). The wiring fact CANNOT be checked "
                        f"here, so THIS RUN MAKES NO 892-C4 CLAIM; see PROVENANCE_AUDIT_903.md "
                        f"for the finding recorded on a checkout where the file was present.")
        return
    try:
        with open(real_leg_path) as fh:
            real_leg = json.load(fh)
    except (OSError, ValueError) as exc:
        audit.fail(gid, f"{real_leg_path} exists but is unreadable/not JSON: {exc}")
        print(f"    [FAIL] 892-C4: calibration_real_leg.json unreadable: {exc}")
        return
    shipped = disclosures_mod.CALIBRATION_TABLE
    shared_blocks = sorted(set(int(k) for k in real_leg) & set(shipped.keys()))
    diffs = [(nb, shipped[nb]["k"], real_leg[str(nb)]["k"])
            for nb in shared_blocks if shipped[nb]["k"] != real_leg[str(nb)]["k"]]
    # This DOES currently differ -- that is the finding, not a bug in this test. Documented,
    # not silently passed: the assertion below FAILS LOUD if a future ship wires the two
    # together (k values become identical) without this audit note being revisited, and FAILS
    # LOUD in the opposite direction if a fresh drift appears at a block count this audit did
    # not see.
    expected_diff_blocks = {2, 3, 4, 5, 6, 10, 20, 80}
    got_diff_blocks = {nb for nb, _, _ in diffs}
    ok = got_diff_blocks == expected_diff_blocks
    print(f"    [{'PASS' if ok else 'FAIL'}] 892-C4: disclosures.CALIBRATION_TABLE (the cheap, "
         f"200/200/400-trial table baked into source and embedded in every real-leg report) "
         f"DIFFERS from runs/892_action_ev/calibration_real_leg.json (the 2000/2000/2000-trial "
         f"table the disclosure calls 'REQUIRED before the real leg launches') at all "
         f"{len(expected_diff_blocks)} shared block counts -- the required higher-precision "
         f"recalibration was computed but never wired into the constant the shipped reports "
         f"actually read. DRIFTS-with-named-cause, sensitivity in PROVENANCE_AUDIT_903.md "
         f"(does NOT flip the ANTI-EDGE band for any of the 3 primary seeds — verified by "
         f"direct recomputation).")
    if not ok:
        audit.fail(gid, f"expected mismatch at block counts {sorted(expected_diff_blocks)}, "
                        f"observed at {sorted(got_diff_blocks)} -- re-check "
                        f"PROVENANCE_AUDIT_903.md, the wiring state changed")
        return
    audit.record(gid, "DRIFTS-with-named-cause — the REQUIRED higher-precision recalibration "
                      "was computed (runs/892_action_ev/calibration_real_leg.json) but never "
                      "wired into disclosures.CALIBRATION_TABLE, which is what every shipped "
                      "report reads; NON-MATERIAL to the D47 ANTI-EDGE band (recomputed)")


# =============================================================================
# 892-C2 / 892-C5 — PROVENANCE-UNRECOVERABLE constants. No command exists to re-run; the gate
# is that the manifest records WHY, not that a re-run occurs.
# =============================================================================
def gate_892c2_c5_unrecoverable_are_disclosed(audit):
    tbl = disclosures_mod.NULL_COVERAGE_TABLE
    tbl_ov = disclosures_mod.NULL_COVERAGE_TABLE_OVERLAPPING
    ok1 = isinstance(tbl, dict) and isinstance(tbl_ov, dict) and len(tbl) >= 6 and len(tbl_ov) >= 6
    print(f"    [{'PASS' if ok1 else 'FAIL'}] 892-C2: NULL_COVERAGE_TABLE/_OVERLAPPING exist and "
         f"are non-trivial ({len(tbl)}/{len(tbl_ov)} rows) but their own comment discloses no "
         f"seed -- PROVENANCE-UNRECOVERABLE, not re-runnable even in principle from the comment "
         f"alone. Sensitivity: NON-MATERIAL — superseded for banding by CALIBRATION_TABLE "
         f"(892-C1, REPRODUCES); this table is carried only as descriptive disclosure + a "
         f"wide-tolerance regression tripwire in test_action_ev_892.py's gate_calibrated_power_bar.")
    if ok1:
        audit.record("892-C2", "PROVENANCE-UNRECOVERABLE — NULL_COVERAGE_TABLE/_OVERLAPPING "
                               "are shipped and non-trivial but disclose no seed, so no re-run "
                               "exists even in principle; NON-MATERIAL (superseded for banding "
                               "by CALIBRATION_TABLE)")
    else:
        audit.fail("892-C2", "NULL_COVERAGE_TABLE/_OVERLAPPING missing or degenerate")

    has_crosscheck = "independent cross-check" in disclosures_mod.CALIBRATION_PRECISION_DISCLOSURE
    print(f"    [{'PASS' if has_crosscheck else 'FAIL'}] 892-C5: the 'independent cross-check "
         f"across three dependence models' claim is present in the shipped disclosure text but "
         f"backed by no script/seed anywhere in the tree (grepped, none found) -- "
         f"PROVENANCE-UNRECOVERABLE. Sensitivity: NON-MATERIAL — used only as prose "
         f"corroboration ('consistent with'), never as a machine-checked constant, and does "
         f"not gate any band.")
    if has_crosscheck:
        audit.record("892-C5", "PROVENANCE-UNRECOVERABLE — the 'independent cross-check across "
                               "three dependence models' claim is shipped in the disclosure "
                               "text with no script or seed anywhere in the tree; NON-MATERIAL "
                               "(prose corroboration, gates no band)")
    else:
        audit.fail("892-C5", "the cross-check disclosure text moved/vanished -- re-locate it")


# =============================================================================
# 892-C6 — ZERO_WIDTH_FLOOR and the MEASURED MARGIN claims in its comment. Cheap and exact:
# three block bootstraps at n_boot=2000, ~2s total. The probe below is the module's OWN shipped
# near-duplicate construction (test_action_ev_892.py:_near_duplicate, the only reconstruction
# the tree supports) at the three per-block-jitter scales the comment names.
# =============================================================================
def _near_duplicate_probe(n_blocks, jitter_std, block=21, base_s=0.02, base_n=0.0, seed=0):
    """test_action_ev_892.py::gate_zero_width_ci_never_bands's ``_near_duplicate``, verbatim in
    construction: per-BLOCK jitter on both sides, repeated ``block`` times within each block."""
    rng = np.random.default_rng(seed)
    s = np.repeat(base_s + rng.normal(0.0, jitter_std, size=n_blocks), block)
    n = np.repeat(base_n + rng.normal(0.0, jitter_std, size=n_blocks), block)
    return s, n


def gate_892c6_zero_width_floor_margins(audit):
    gid = "892-C6"
    floor = report_mod.ZERO_WIDTH_FLOOR
    ok_floor = floor == 1e-4
    print(f"    [{'PASS' if ok_floor else 'FAIL'}] 892-C6: report.ZERO_WIDTH_FLOOR == {floor!r} "
         f"(1 basis point, the value its comment argues for) — the CONSTANT ITSELF REPRODUCES")
    if not ok_floor:
        audit.fail(gid, f"ZERO_WIDTH_FLOOR is {floor!r}, comment argues for 1e-4")

    # (a) THE LOAD-BEARING CLAIM: homogeneity of degree 1 -- ci_width / jitter_std constant
    # across decades. This is the claim the whole "no self-referential statistic can do this
    # job, you need an EXTERNAL anchor" argument rests on, and it is the reason the floor is an
    # absolute number rather than a ratio.
    ratios = {}
    for jitter in (1e-9, 1e-6, 1e-4):
        s, n = _near_duplicate_probe(35, jitter, seed=0)
        ci = report_mod.block_bootstrap_delta_ci(s, n, block=21, n_boot=2000, seed=0)
        ratios[jitter] = ci["ci_width"] / jitter
    spread = max(ratios.values()) - min(ratios.values())
    ok_homog = spread < 1e-6                     # identical to ~7 significant figures, measured
    print(f"    [{'PASS' if ok_homog else 'FAIL'}] 892-C6: ci_width/jitter_std at per-block "
         f"jitter 1e-9 / 1e-6 / 1e-4 = "
         f"{' / '.join(f'{ratios[j]:.6f}' for j in (1e-9, 1e-6, 1e-4))} (spread {spread:.2g}) — "
         f"the comment's load-bearing homogeneity-of-degree-1 claim REPRODUCES exactly, so no "
         f"scale-relative statistic can separate the cases and an external anchor is required")
    if not ok_homog:
        audit.fail(gid, f"homogeneity claim DRIFTS -- ci_width/jitter varies across "
                        f"decades: {ratios}")

    # (b) THE COVERED SCALES still band as the guard intends (the behaviour anything downstream
    # could actually depend on), at both block counts the shipped test exercises.
    bad = []
    for jitter, n_blocks in ((1e-9, 35), (1e-6, 35), (1e-9, 33), (1e-6, 33)):
        s, n = _near_duplicate_probe(n_blocks, jitter, seed=int(jitter * 1e12) + n_blocks)
        ci = report_mod.block_bootstrap_delta_ci(s, n, block=21, n_boot=2000, seed=0)
        if not (ci["zero_width_ci"] and report_mod.band_for(ci) == report_mod.BAND_ZERO_WIDTH):
            bad.append((jitter, n_blocks, ci["ci_width"], report_mod.band_for(ci)))
    ok_bands = not bad
    print(f"    [{'PASS' if ok_bands else 'FAIL'}] 892-C6: at 1e-9 and 1e-6 per-block jitter "
         f"(n_blocks 33 and 35) the emitted CI still reads ZERO-WIDTH-CI and is excluded from "
         f"banding — REPRODUCES, and matches the lane's existing executable coverage at "
         f"test_action_ev_892.py:1474 gate_zero_width_ci_never_bands")
    if not ok_bands:
        audit.fail(gid, f"the zero-width guard stopped firing at a covered scale: {bad}")

    # (c) THE DRIFT. Two of the comment's stated MEASURED MARGINs are one order of magnitude
    # too generous, and they are arithmetically incompatible with the comment's OWN homogeneity
    # claim in (a) above: with ci_width/jitter_std pinned at ~1.06, a 1e-6 jitter emits O(1e-6),
    # not the advertised "O(1e-7) -- three orders below"; and the closing parenthetical's 1e-4
    # case ("its emitted width is O(1e-5), below this floor, so it reads ZERO-WIDTH-CI here --
    # deliberately") is wrong in BOTH its number and its conclusion: the width is O(1e-4) and
    # whether it lands above or below the floor is a coin flip in the DATA seed (measured: 4 of
    # 8 seeds above). The comment mis-transcribes its own measurement; the code is consistent
    # with itself throughout. Pinned here so the mis-statement cannot be quietly "fixed" in the
    # comment without this audit note being revisited (D39: superseded records stay visible).
    w_1e6 = ratios[1e-6] * 1e-6
    w_1e4 = ratios[1e-4] * 1e-4
    drift_1e6 = w_1e6 > 1e-6 * 0.3          # advertised O(1e-7); measured O(1e-6)
    drift_1e4 = w_1e4 > floor               # advertised "reads ZERO-WIDTH-CI"; measured: above
    ok_drift = drift_1e6 and drift_1e4
    print(f"    [{'PASS' if ok_drift else 'FAIL'}] 892-C6: the comment's MEASURED MARGIN prose "
         f"DRIFTS-with-named-cause — advertised width at 1e-6 jitter 'O(1e-7)', measured "
         f"{w_1e6:.3g} (O(1e-6)); advertised 1e-4-jitter case 'O(1e-5) ... reads ZERO-WIDTH-CI', "
         f"measured {w_1e4:.3g} > floor {floor:g} and bands EDGE at this seed. Named cause: the "
         f"margins were transcribed an order too small and contradict the comment's own "
         f"(reproducing) homogeneity claim. NON-MATERIAL: no code path reads this prose, the "
         f"floor value and the guard's behaviour at every covered scale are unaffected, and no "
         f"892 band or the D47 ANTI-EDGE verdict depends on it.")
    if not ok_drift:
        audit.fail(gid,
            f"the drift this audit named has itself changed (1e-6 width {w_1e6:.3g}, "
            f"1e-4 width {w_1e4:.3g} vs floor {floor:g}) -- re-derive PROVENANCE_AUDIT_903.md's "
            f"892-C6 section, either the comment or the estimator moved")

    if ok_floor and ok_homog and ok_bands and ok_drift:
        audit.record(gid, "ZERO_WIDTH_FLOOR = 1e-4 and its load-bearing homogeneity claim "
                          "REPRODUCE (ratio spread 2.7e-09 across three decades); two MEASURED "
                          "MARGIN prose values DRIFT-with-named-cause (transcribed an order too "
                          "small), NON-MATERIAL — no code path reads the prose")


# =============================================================================
# 837-C1 — the fixture-smoke document. Cheap, fully local, re-run through the real subcommand.
#
# CYCLE-2 REBUILD — BLOCKER. The cycle-1 version of this gate ran `walkforward.py smoke` IN THE
# LIVE WORKING TREE and then "self-cleaned" by passing whatever `git diff --name-only --
# experiments/837_walkforward` returned AFTERWARDS to `git checkout --`. That mechanism cannot
# distinguish "dirtied by this run" from "was already dirty for an unrelated reason": an
# uncommitted edit to a file walkforward.py never writes (demonstrated on 837's SPEC.md) was
# silently destroyed at exit 0 with no warning, and the `finally` block reverted even on runs
# where the smoke subprocess itself failed rc=1. That is the run-34 incident class (a stray
# `git stash` stripping a concurrent lane's work) baked into a committed test that every future
# run would execute. The stakes were not cosmetic: `run_smoke` (walkforward.py:1147) calls
# `materialize(FIXTURE_DB, REAL_FOLDS, "real", ...)` unconditionally, so the smoke overwrites
# the TRACKED real-fold definitions that gate real training/eval launches -- which made that
# fragile revert the only thing standing between a smoke run and a poisoned `folds/real/`.
#
# THE REBUILD, and why it is safe:
#   * The live working tree is NEVER WRITTEN TO and NEVER REVERTED. The smoke re-run happens
#     inside a throwaway detached `git worktree` checked out at HEAD in a temp directory that is
#     CHECKED to lie outside the repo. Every file walkforward.py rewrites is rewritten THERE.
#     There is no restore step, because nothing in the live tree is ever dirtied -- so there is
#     no restore step that can fail half-way, revert the wrong bytes, or fire on the wrong exit
#     path.
#   * NO `git checkout`, `git stash`, `git reset`, `git add`, or `git worktree prune` is issued
#     anywhere in this module. Against the live repo it runs only `worktree add` / `worktree
#     remove --force` (administrative, path-scoped) and read-only `status`/`log`.
#   * FAIL CLOSED. If `experiments/837_walkforward/` is not clean in the live tree BEFORE
#     anything happens, the gate refuses: it skips loudly, names the offending paths, records
#     the skip so the run exits non-zero (2) and claims NOTHING about this constant, and does
#     nothing else.
#   * ON SIGKILL (or SIGTERM, or a hard crash, or a smoke subprocess that dies): the live
#     working tree is byte-for-byte unchanged, because no code path in this module writes
#     outside the throwaway directory. What leaks is the throwaway directory itself plus its
#     administrative entry under `.git/worktrees/` -- bookkeeping only, no tracked file. It is
#     NOT reclaimed here: see the teardown comment for why this module refuses to run
#     `git worktree prune`.
#   * experiments/897_corrected_809/ (a concurrent lane's live uncommitted work) is never read,
#     listed, staged, or touched: the only paths this module names are under WF_DIR, EXP_DIR,
#     runs/892_action_ev/, and the throwaway temp dir.
#
# The isolation is DEMONSTRATED, not asserted: `test_903_worktree_isolation_demo.py` beside this
# file plants a decoy uncommitted edit under experiments/837_walkforward/ inside its OWN
# disposable worktree and shows the cycle-1 mechanism destroying it and this one preserving it.
# =============================================================================
_KNOWN_837_DRIFT_SUBSTRINGS = (
    # TASK_869 (2026-08-03) fixed a UTC/DST off-by-one-hour bug in the eval-window start
    # timestamp used in this negative-test's OWN rejection message; results_fixture_smoke.json
    # was generated 2026-07-08, before that fix, and was never regenerated after it landed. The
    # smoke artifact's own verdict text says "plumbing proof only, no verdict" -- NON-MATERIAL.
    ("1761951600", "1761955200"),
    ("vectorbt_env.py:584", "vectorbt_env.py:660"),
)


class ThrowawaySetupError(RuntimeError):
    """The disposable worktree could not be set up SAFELY. Always a SKIP, never a silent pass:
    it means the gate could not be run, not that the constant reproduced."""


def live_wf_dirty_paths():
    """Porcelain status of the LIVE working tree, scoped to WF_DIR. Read-only: `git status`
    writes no tracked file. Returns the raw porcelain lines (empty list == clean).

    Raises on a git failure rather than returning ``[]`` -- "git broke" must never be
    indistinguishable from "the tree is clean". Callers turn the exception into a SKIP.
    """
    rel = os.path.relpath(WF_DIR, REPO)
    out = subprocess.run(["git", "status", "--porcelain", "--", rel],
                         cwd=REPO, capture_output=True, text=True, check=True).stdout
    return [ln for ln in out.splitlines() if ln.strip()]


def _throwaway_dir(prefix):
    """``tempfile.mkdtemp`` that REFUSES to hand back a directory inside the repo.

    mkdtemp honours ``$TMPDIR``; with TMPDIR pointing somewhere inside the working tree, the
    "throwaway" worktree would materialise INSIDE the tree this module promises never to write
    to -- and on SIGKILL it would leak a ~23MB nested checkout that 837-C1's post-condition
    (scoped to experiments/837_walkforward/) could not see. Refuse instead of writing there.
    """
    wt = tempfile.mkdtemp(prefix=prefix)
    real_wt = os.path.realpath(wt)
    real_repo = os.path.realpath(REPO)
    if real_wt == real_repo or real_wt.startswith(real_repo + os.sep):
        os.rmdir(wt)
        raise ThrowawaySetupError(
            f"tempfile.mkdtemp() resolved to {real_wt}, which is INSIDE the repo "
            f"({real_repo}) -- $TMPDIR={os.environ.get('TMPDIR')!r}. Refusing to create a "
            f"throwaway worktree inside the working tree this module must not write to. "
            f"Re-run with TMPDIR pointing outside the repo.")
    return wt


def smoke_in_throwaway_worktree():
    """Run the advertised ``walkforward.py smoke`` at HEAD inside a DISPOSABLE detached git
    worktree in a temp dir. Returns ``(returncode, stdout, results_fixture_smoke_dict_or_None)``;
    ``returncode is None`` means the gate could not be RUN (setup failed, smoke timed out, the
    output was unreadable) -- the caller must turn that into a SKIP, never a pass.

    WHY A WORKTREE (this function is the cycle-2 BLOCKER fix): every file the smoke rewrites is
    rewritten inside the throwaway directory, so the live tree is never dirtied and therefore
    never needs reverting. See the block comment above for the full failure argument.

    NOT REACHABLE FROM HERE, BY CONSTRUCTION: no `git checkout`, `git stash`, `git reset`,
    `git add`, or `git worktree prune` is ever issued. The only git commands run against the
    live repo are ``worktree add`` / ``worktree remove --force`` (administrative, and BOTH
    name the throwaway path explicitly) plus read-only ``status``/``log`` elsewhere in this
    module.
    """
    try:
        wt = _throwaway_dir("task903_wf_smoke_")
    except (ThrowawaySetupError, OSError) as exc:
        return None, f"throwaway directory setup failed: {exc}", None
    added = False
    try:
        os.rmdir(wt)                  # `git worktree add` insists on creating the dir itself
        add = subprocess.run(["git", "worktree", "add", "--detach", wt, "HEAD"],
                             cwd=REPO, capture_output=True, text=True)
        if add.returncode != 0:
            return None, f"git worktree add failed: {add.stderr[-400:]}", None
        added = True
        # The fixture db is gitignored, so a fresh worktree has no copy of it. SYMLINK it in
        # (76MB; never copied, never written): walkforward.py:220 opens every db handle
        # `file:...?mode=ro`, and the fixture is journal_mode=delete, so sqlite creates no
        # -wal/-shm sidecar next to the live file. trading_data.db is never referenced.
        wt_fixtures = os.path.join(wt, "fixtures")
        os.makedirs(wt_fixtures, exist_ok=True)
        os.symlink(os.path.join(REPO, "fixtures", "trading_fixture.db"),
                   os.path.join(wt_fixtures, "trading_fixture.db"))
        proc = subprocess.run(
            [sys.executable, os.path.join(wt, "experiments", "837_walkforward",
                                          "walkforward.py"), "smoke"],
            cwd=wt, capture_output=True, text=True, timeout=600)
        fresh = None
        fresh_path = os.path.join(wt, "experiments", "837_walkforward",
                                  "results_fixture_smoke.json")
        if os.path.exists(fresh_path):
            with open(fresh_path) as fh:
                fresh = json.load(fh)
        return proc.returncode, proc.stdout, fresh
    except subprocess.TimeoutExpired as exc:
        return None, (f"`walkforward.py smoke` exceeded its {exc.timeout}s timeout in the "
                      f"throwaway worktree and was killed"), None
    except (OSError, ValueError) as exc:
        return None, (f"the throwaway worktree could not be prepared, or its smoke output "
                      f"could not be read: {type(exc).__name__}: {exc}"), None
    finally:
        # Teardown. Both branches are non-destructive to the live tree AND to every other
        # worktree in this repo:
        #   * `git worktree remove --force <path>` is PATH-SCOPED and already returns 0 when
        #     the directory has vanished, so it covers the crash case on its own.
        #   * There is deliberately NO `git worktree prune` fallback (cycle-3 MINOR 3). `prune`
        #     takes no path filter: it deletes the administrative record of ANY worktree whose
        #     directory it cannot see at that instant -- including another lane's LIVE,
        #     unlocked worktree that happens to be momentarily absent (demonstrated: index,
        #     HEAD and reflog all deleted). A stale bookkeeping entry here is harmless; eating
        #     a sibling lane's worktree record is not. Leave the record; a human can prune.
        if added:
            rm = subprocess.run(["git", "worktree", "remove", "--force", wt],
                                cwd=REPO, capture_output=True, text=True)
            if rm.returncode != 0:
                shutil.rmtree(wt, ignore_errors=True)
                print(f"    [warn] `git worktree remove --force {wt}` failed "
                      f"({rm.stderr.strip()[-200:]}); the directory was deleted and a stale "
                      f"bookkeeping entry may remain under .git/worktrees/. NOT pruning: "
                      f"prune is repo-wide and would risk other lanes' worktrees.")
        else:
            shutil.rmtree(wt, ignore_errors=True)


def gate_837c1_fixture_smoke_drift_is_named(audit):
    gid = "837-C1"
    # --- FAIL-CLOSED PRE-CHECK, before anything at all happens ------------------------------
    try:
        dirty = live_wf_dirty_paths()
    except (subprocess.SubprocessError, OSError) as exc:
        audit.skip(gid, f"could not establish whether {REPO} is clean under "
                        f"{os.path.relpath(WF_DIR, REPO)} (`git status` failed: {exc}). "
                        f"Refusing to run a side-effecting command against a tree whose state "
                        f"is unknown. Nothing was run and nothing was touched.")
        return
    if dirty:
        audit.skip(gid,
                   f"the working tree this gate runs against ({REPO}) is NOT clean under "
                   f"{os.path.relpath(WF_DIR, REPO)} — {dirty}. Nothing was run and nothing "
                   "was touched. Two reasons to refuse rather than proceed: (1) somebody is "
                   "working there and this gate will not gamble with their uncommitted bytes, "
                   "and (2) the shipped-vs-fresh comparison would not be against the COMMITTED "
                   "artifact anyway, so a result would be meaningless. Commit or park that "
                   "work, then re-run this file.")
        return

    shipped_path = os.path.join(WF_DIR, "results_fixture_smoke.json")
    try:
        with open(shipped_path) as fh:               # read-only; verified == HEAD above
            shipped = json.load(fh)
    except (OSError, ValueError) as exc:
        audit.fail(gid, f"the enumerated artifact {shipped_path} is missing or unreadable "
                        f"({type(exc).__name__}: {exc}) -- the manifest points at a tracked "
                        f"file that must exist")
        print(f"    [FAIL] 837-C1: results_fixture_smoke.json missing/unreadable: {exc}")
        return

    rc, stdout, fresh = smoke_in_throwaway_worktree()
    if rc is None:
        audit.skip(gid, f"the re-run could not be performed at all: {stdout}. No comparison was "
                        f"made, so THIS RUN MAKES NO 837-C1 CLAIM. The live tree was not "
                        f"written to.")
        return
    ok_ran = rc == 0 and "SMOKE PASS" in (stdout or "") and fresh is not None
    print(f"    [{'PASS' if ok_ran else 'FAIL'}] 837-C1: 'python walkforward.py smoke' (the "
         f"advertised command), run at HEAD inside a throwaway git worktree, exits 0 and "
         f"reports SMOKE PASS")
    if not ok_ran:
        audit.fail(gid, f"walkforward.py smoke did not pass in the throwaway worktree: "
                        f"rc={rc} stdout_tail={(stdout or '')[-400:]!r}")
        return

    # --- POST-CONDITION: prove the live tree is still exactly as we found it ----------------
    try:
        still_clean = live_wf_dirty_paths()
    except (subprocess.SubprocessError, OSError) as exc:
        audit.fail(gid, f"the post-condition could not be checked (`git status` failed after "
                        f"the re-run: {exc}) -- verify by hand that {WF_DIR} is clean")
        print(f"    [FAIL] 837-C1: post-condition `git status` failed: {exc}")
        return
    ok_clean = not still_clean
    print(f"    [{'PASS' if ok_clean else 'FAIL'}] 837-C1: the LIVE working tree is still clean "
         f"under {os.path.relpath(WF_DIR, REPO)} after the re-run — the smoke's writes "
         f"(results_fixture_smoke.json + the 7 re-materialized real-fold definition files) all "
         f"landed in the throwaway worktree, and NOTHING was reverted in the live tree because "
         f"nothing in it was ever dirtied")
    if not ok_clean:
        audit.fail(gid, f"the live tree was dirtied under WF_DIR despite the worktree "
                        f"isolation: {still_clean} -- STOP, the isolation is broken; do NOT "
                        f"revert these by hand without checking whose they are")
        return

    a, b = dict(shipped), dict(fresh)
    a.pop("generated", None)
    b.pop("generated", None)
    # smoke_aggregate.ledger_row embeds a SECOND run-date stamp ("| 837/R4 | YYYY-MM-DD | ...")
    # inside a prose string -- the same volatility as the top-level "generated" field, just not
    # a top-level key, so it needs its own normalization rather than a dict .pop().
    date_re = re.compile(r"(\| 837/R4 \| )\d{4}-\d{2}-\d{2}( \|)")
    for d in (a, b):
        agg = d.get("smoke_aggregate")
        if isinstance(agg, dict) and isinstance(agg.get("ledger_row"), str):
            agg["ledger_row"] = date_re.sub(r"\1<RUN-DATE>\2", agg["ledger_row"])
    sa, sb = json.dumps(a, indent=2, sort_keys=True), json.dumps(b, indent=2, sort_keys=True)
    if sa == sb:
        print("    [PASS] 837-C1: results_fixture_smoke.json is BIT-IDENTICAL to a fresh "
             "re-run (excl. the 'generated' timestamp) — REPRODUCES")
        audit.record(gid, "REPRODUCES — results_fixture_smoke.json is bit-identical to a fresh "
                          "`walkforward.py smoke` (excl. the 'generated' timestamps)")
        return
    # NORMALIZE ONLY AT THE KNOWN LOCUS (cycle-3 MINOR 5). A whole-document `sa.replace(old,
    # new)` would silently rewrite a SECOND, unexamined occurrence of the same substring -- in
    # a detector whose entire job is to not do that. Both substrings occur exactly once today;
    # assert that, and refuse (loudly) rather than normalize if a future document repeats them.
    for old, new in _KNOWN_837_DRIFT_SUBSTRINGS:
        n_old, n_new = sa.count(old), sb.count(new)
        if n_old == 0 and n_new == 0:
            continue                              # this known drift is no longer present at all
        if n_old != 1 or n_new != 1:
            audit.fail(gid, f"the known TASK_869 drift pair {old!r} -> {new!r} occurs {n_old}x "
                            f"in the shipped document and {n_new}x in the fresh one, not "
                            f"exactly once each. REFUSING to normalize: a whole-document "
                            f"replace would rewrite a locus this audit never inspected. "
                            f"Re-derive PROVENANCE_AUDIT_903.md's 837-C1 section by hand.")
            print(f"    [FAIL] 837-C1: known-drift substring {old!r} is not single-locus "
                  f"({n_old}x shipped / {n_new}x fresh) -- refusing a blanket replace")
            return
        sa = sa.replace(old, new, 1)              # count==1 asserted above: locus-exact
    if sa == sb:
        print("    [PASS] 837-C1: results_fixture_smoke.json DRIFTS from a fresh re-run only at "
             "the KNOWN, NAMED, dated cause (TASK_869 2026-08-03 UTC-windows fix corrected a "
             "stale 1-hour timestamp in a negative-test rejection message) — "
             "DRIFTS-with-named-cause, NON-MATERIAL (the artifact's own verdict text says "
             "'plumbing proof only, no verdict')")
        audit.record(gid, "DRIFTS-with-named-cause — results_fixture_smoke.json differs from a "
                          "fresh `walkforward.py smoke` only at the single, dated TASK_869 "
                          "(2026-08-03) UTC-windows locus; NON-MATERIAL (the artifact is a "
                          "plumbing proof carrying stub scorecards and no verdict)")
    else:
        audit.fail(gid, "results_fixture_smoke.json drifts from a fresh re-run at a location "
                        "NOT covered by the known TASK_869 cause -- new, unexplained drift, "
                        "escalate")
        print("    [FAIL] 837-C1: unexplained drift beyond the known TASK_869 cause")


# =============================================================================
# 875-C1 — the walk-forward real numbers. THIS GATE IS HOST-DEPENDENT BY NATURE and says so:
# `runs/875_wf/` is gitignored and lives on gpu-server, so the honest answer differs by where
# you run. An earlier revision hardcoded "unreachable from the workstation" as THE finding and
# FAILED loudly the first time it ran on srv (correctly — the access picture had changed). The
# lesson kept here: a provenance gate must classify what it can SEE, not assert a fixed verdict.
# =============================================================================
def gate_875c1_walkforward_numbers(audit):
    gid = "875-C1"
    repo_root = REPO
    present = [rel for rel in ("runs/875_wf", "runs/875_wf/grade_correct.py")
               if os.path.exists(os.path.join(repo_root, rel))]

    if not present:
        # --- artifacts not on this host: verify the absence, claim nothing about the numbers ---
        log = subprocess.run(
            ["git", "log", "--all", "--oneline", "--", "runs/875_wf", "*orch_wf875*"],
            cwd=repo_root, capture_output=True, text=True)
        if log.returncode != 0:
            audit.skip(gid, f"`git log --all -- runs/875_wf '*orch_wf875*'` failed "
                            f"({log.stderr.strip()[-200:]}); the absence-from-history half "
                            f"could not be established, so no claim is made.")
            return
        if log.stdout.strip():
            audit.fail(gid, f"expected zero git history for runs/875_wf/orch_wf875 -- "
                            f"found: {log.stdout.strip()!r}")
            return
        print(f"    [SKIP] 875-C1: runs/875_wf/ is verified ABSENT from THIS host's tree and "
              f"from all of git history (gitignored, srv-only). The LEDGER's "
              f"875/wf-generalization numbers cannot be re-read here, so this run makes NO "
              f"provenance claim about them. They ARE re-derivable — cheaply, no training — on "
              f"gpu-server: `python3 runs/875_wf/grade_correct.py`. Run this gate there.")
        audit.skip(gid, "runs/875_wf/ absent on this host (verified, incl. git history); the "
                        "numbers are re-derivable on gpu-server via grade_correct.py -- no "
                        "provenance claim made from here")
        return

    # --- artifacts present (gpu-server): actually ASK the question ---
    grader = os.path.join(repo_root, "runs/875_wf/grade_correct.py")
    if not os.path.exists(grader):
        audit.skip(gid, "runs/875_wf/ exists but grade_correct.py does not -- cannot re-derive")
        return
    r = subprocess.run([sys.executable, grader], cwd=repo_root,
                       capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        audit.fail(gid, f"grade_correct.py exited {r.returncode}: {r.stderr.strip()[-300:]}")
        return
    out = r.stdout
    # The LEDGER row's per-fold medians and floors, transcribed from experiments/LEDGER.md.
    expected = [("+1.4957", "+3.8987"), ("+0.3995", "+1.4301"), ("+0.3108", "+1.3952"),
                ("+0.4121", "+0.2635"), ("-2.9566", "+0.4244")]
    drift = [f"fold {i + 1}: MEDIAN={m} floor={f} not found in the re-grade"
             for i, (m, f) in enumerate(expected)
             if f"MEDIAN={m}  median_floor={f}" not in out]
    if "folds graded: 5/5" not in out or "above own floor: 1" not in out:
        drift.append("the 5/5-graded, 1-above-floor tally did not reproduce")
    if drift:
        audit.fail(gid, "875 walk-forward numbers DRIFT from the LEDGER row: " + "; ".join(drift))
        return
    print(f"    [PASS] 875-C1: re-derived on this host from the raw per-seed JSONs via "
          f"`grade_correct.py` (read-only, no training, no re-run of the 8h57m battery) — all "
          f"five folds' MEDIAN and floor, and the 'folds graded: 5/5 | above own floor: 1' "
          f"tally, reproduce the LEDGER's 875/wf-generalization row EXACTLY. REPRODUCES. "
          f"NOTE the shipped summary artifact is NOT the source: WF_SUMMARY.json is `[]`, a "
          f"documented silent fake-null (runs/875_wf/NOTE_summary_key_mismatch.md — "
          f"orch_wf875.sh read `sharpe`/`teacher.floor_inenv`, the schema keys are "
          f"`sharpe_full`/`tiers['21']['teacher_floor_inenv']`), so anyone reading the "
          f"canonical-looking summary gets nothing. The numbers live ONLY in the raw per-seed "
          f"results JSONs plus grade_correct.py.")
    audit.record(gid, "REPRODUCES — all five folds' medians/floors and the 1-of-5-above-floor "
                      "tally re-derived exactly from the raw per-seed JSONs on gpu-server; the "
                      "'no OOS edge' verdict rests on numbers that reproduce. Caveat recorded: "
                      "WF_SUMMARY.json ships as `[]` (documented fake-null), so the summary "
                      "artifact is not the provenance source -- grade_correct.py is")


def _run_gate(audit, gate_id, fn):
    """Run one gate, and make sure it can never vanish silently. An unhandled exception used to
    propagate out of main() as a traceback that skipped every LATER gate (cycle-3 MINOR 4);
    it is now a named failure against this gate, and the remaining gates still run."""
    try:
        fn(audit)
    except Exception as exc:                                          # noqa: BLE001 - deliberate
        print(f"    [FAIL] {gate_id}: unhandled {type(exc).__name__}: {exc}")
        audit.fail(gate_id, f"the gate crashed with an unhandled {type(exc).__name__}: {exc} "
                            f"-- no classification was established")


def main():
    audit = Audit()
    print("=" * 84)
    print("[903] TASK_903 constant-provenance audit — 892 calibration table + 875 walk-forward")
    print(f"[903] {EXIT_CODE_CONTRACT}")
    print("=" * 84)

    print("[903] gate (i) — the enumeration itself is checkable")
    _run_gate(audit, "gate-(i)", gate_enumeration_complete)

    print("[903] 892-C1 — CALIBRATION_TABLE vs its advertised command")
    _run_gate(audit, "892-C1", gate_892c1_calibration_table_reproduces)

    print("[903] 892-C3 — SPEC.md body sha256 provenance stamp")
    _run_gate(audit, "892-C3", gate_892c3_spec_hash_reproduces)

    print("[903] 892-C4 — the required-but-unwired higher-precision recalibration")
    _run_gate(audit, "892-C4", gate_892c4_real_leg_calibration_not_wired_in)

    print("[903] 892-C2 / 892-C5 — PROVENANCE-UNRECOVERABLE constants, disclosed not silent")
    _run_gate(audit, "892-C2/C5", gate_892c2_c5_unrecoverable_are_disclosed)

    print("[903] 892-C6 — ZERO_WIDTH_FLOOR and its in-comment MEASURED MARGIN claims")
    _run_gate(audit, "892-C6", gate_892c6_zero_width_floor_margins)

    print("[903] 837-C1 — results_fixture_smoke.json vs its advertised command "
          "(run in a throwaway worktree; the live tree is never written to)")
    _run_gate(audit, "837-C1", gate_837c1_fixture_smoke_drift_is_named)

    print("[903] 875-C1 — the walk-forward real numbers, verified unreachable from here")
    _run_gate(audit, "875-C1", gate_875c1_walkforward_numbers)

    # No declared gate may end the run without a finding, a skip, or a failure. (The C2/C5 pair
    # shares one function but records under both ids, so both must show up.)
    accounted = audit.accounted()
    for gate_id in DECLARED_GATES:
        if gate_id not in accounted:
            audit.fail(gate_id, "the gate reached NO conclusion this run (it neither recorded a "
                                "finding, nor skipped, nor failed) -- treat the whole run as "
                                "unreliable and investigate")

    print("=" * 84)
    print(f"[903] {EXIT_CODE_CONTRACT}")

    if audit.skipped:
        print(f"[903] {len(audit.skipped)} gate(s) SKIPPED (fail-closed — NOT a pass, and NOT "
              f"claimed below): {', '.join(g for g, _ in audit.skipped)}")
        for gate_id, reason in audit.skipped:
            print(f"  - {gate_id}: {reason}")

    if audit.failures:
        print(f"[903] FAIL ({EXIT_FAILED}) — {len(audit.failures)} check(s) failed:")
        for gate_id, detail in audit.failures:
            print(f"  - {gate_id}: {detail}")
        if audit.findings:
            print(f"[903] (findings this run DID establish, for context: "
                  f"{', '.join(g for g, _ in audit.findings)})")
        return EXIT_FAILED

    # THE SUMMARY IS BUILT FROM WHAT RAN — never a hardcoded list of all classifications.
    print(f"[903] FINDINGS ESTABLISHED BY THIS RUN ({len(audit.findings)} of "
          f"{len(DECLARED_GATES)} declared gates):")
    for gate_id, classification in audit.findings:
        print(f"  - {gate_id}: {classification}")

    if audit.skipped:
        print(f"[903] INCOMPLETE ({EXIT_INCOMPLETE}) — nothing failed, but "
              f"{len(audit.skipped)} gate(s) did not run "
              f"({', '.join(g for g, _ in audit.skipped)}). This run establishes NOTHING about "
              f"them: do not quote a classification for a skipped gate from this output. "
              f"Resolve the skip conditions listed above and re-run for a complete audit.")
        return EXIT_INCOMPLETE

    print(f"[903] PASS ({EXIT_OK}) — all {len(DECLARED_GATES)} declared gates ran to a "
          f"conclusion, {len(audit.findings)} findings recorded above, 0 skipped, 0 failed.")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
