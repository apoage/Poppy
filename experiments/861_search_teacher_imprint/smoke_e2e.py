#!/usr/bin/env python3
"""TASK_861 — end-to-end fixture smoke for the search-teacher imprint (FACTORY half).

Chains the two flag-gated extensions on the fixture and proves the four acceptance gates:

  (i)   CANON PINS — both flags OFF is bit-identical.
        * 835: the fixture smoke reproduces the CHECKED-IN
          ``experiments/835_stage4_eval/results_fixture_smoke.json`` exactly (every key except
          the run-pointer/ledger fields, which carry the out-dir and today's date).
        * 832: the default fixture smoke reproduces the PRE-CHANGE scored values pinned in
          ``CANON_832_PINS`` below (captured from the tree before this task's diff, float-exact).
  (i-b) WEIGHT-ON PARITY PIN (TASK_866) — the 861 pins above are a weight-OFF oracle while the
        committed 832 baseline is the weight-ON deliverable, so nothing pinned the weight-ON
        numbers (the 864 incident class). This leg runs the CANON weight-ON recipe and pins it
        against the tracked ``experiments/832_linear_imprint/results_fixture_smoke.json`` READ
        AS THE ORACLE: fail-CLOSED when that file is missing / git-dirty / not itself the
        weight-ON order-code configuration, and a cross-configuration comparison (weight-OFF
        artifact vs weight-ON oracle or vice versa) is a REFUSAL with the mismatch NAMED,
        never a numeric diff. Both mutation directions are demonstrated in-leg.
  (ii)  COLLECTION IS A PURE RECORDER — the identical eval command run with and without
        ``--collect-chosen-vectors`` writes a BYTE-IDENTICAL scorecard JSON.
  (iii) PROVENANCE FAIL-LOUD — a mismatched window, a mismatched panel frame (universe) and a
        mismatched env-flag block are each REFUSED (non-zero exit, named field, nothing trained).
  (iv)  FALSIFICATION, SEED-SWEPT — the legit imprint clears its gate on every seed while the
        ``--shuffle-teacher-vectors`` sabotage FAILS on every seed (the 860 fail-closed pattern).
  (v)   PINNED-RECIPE CO-RUNNABILITY — the two pre-registered real legs run TOGETHER: a
        collection recorded at the teacher of record's ``position_size_pct`` 0.1 imprints under
        the pinned ``jobs_imprint_full.manifest`` recipe (``--imprint-mapping weight-native``,
        this runner's own 0.5/n sizing) and exits 0.
  (vi)  MINUTE FRAME PIN (TASK_865) — the SAME chain on the SAME minute panel with the universe
        PINNED instead of coverage-ranked (``--panel-frame-source minute``): collect -> imprint ->
        sanity replay exits 0 with the frame/inactive-slot disclosure in the artifact, and a
        same-membership REORDER of that frame is REFUSED. The exhaustive minute-pin matrix
        (identity scrambling, source gate, degeneracy detector, real-DB coverage) lives in
        ``experiments/865_minute_frame_pin/smoke_minute_frame_pin.py``; this leg is the 861
        chain's own coverage of the path its real legs will run on.

FIX-PASS legs (the adversarial panel's MAJOR findings, each with a fail-closed demonstration):
  * OFF-PIN TEACHER — a collection recorded with ``sims`` 8 instead of the pinned 32 is REFUSED
    (FIX 1); the escape hatch ``--allow-off-pin-teacher`` runs it but stamps
    ``operator_off_pin: true``. HONESTY PIN: the smoke's own legit runs must be off-pin in
    EXACTLY ONE field, ``checkpoint`` (it collects under a checkpoint it just minted) — every
    operator knob must still match the +8.024 pin.
  * PAYLOAD INTEGRITY — a byte-edited payload fails the sha256 digest; a payload rewritten as
    non-integer codes (with the digest recomputed) is caught by the integrality check; a payload
    PERMUTED with the digest recomputed — which passes every provenance, pin and integrality
    check — is caught by the recorded-walk-score replay assertion (FIX 2).
  * EMPTY-STRING FLAGS fail CLOSED (FIX 3): ``--collect-chosen-vectors ""`` and
    ``--teacher-vectors ""`` both exit non-zero, before any walk, and the checked-in 832 parity
    baseline is left untouched. An unwritable collection path is rejected BEFORE the walk.
  * CONDITIONAL ENV EXEMPTION (FIX 4): ``position_size_pct`` is exempt only while
    ``vector_weight_rebalance`` holds on both sides — with weight semantics OFF the same
    mismatch is REFUSED.

Plus a round-trip cross-check: the imprint's in-env teacher floor at net@21 must reproduce the
collection walk's own net@21 Sharpe — the same actions, the same rows, the same env, walked by
two different modules. (The runner now asserts this internally too; this leg re-derives it from
the artifacts.)

  (vii) TRACKED-BASELINE PINS (TASK_867) — the two COMMITTED baselines of this module
        (``chosen_vectors_fixture.json`` and ``results_fixture_smoke.json``) are pinned against
        what the current code + recipe reproduces, minus documented volatile fields. Default
        invocations write UNTRACKED ``*_local.json`` outputs, so no ordinary verification run
        can dirty (or, under ``--quick``, silently DOWNGRADE) a tracked baseline; overwriting
        one requires the explicit ``--promote-baselines`` flag — recipe-pinned (full-seed
        canon re-derived from argparse defaults), validity-gated, atomic (``.tmp`` +
        ``os.replace``), dirty-refusing, per-field diff printed. This is the shipped TASK_864
        promote gate (``imprint.py``) ported to this module's two baselines.

CPU / fixture only; no training spend beyond the ~80-iteration smoke imprints. NEVER touches
trading_data.db. Run: ``python experiments/861_search_teacher_imprint/smoke_e2e.py``
(``--quick`` runs one seed per falsification arm instead of four). Both invocations leave the
tracked baselines untouched; ``--promote-baselines`` is the ONLY way they move.
"""
import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(EXP_DIR))
RUN_DIR = os.path.join(EXP_DIR, "smoke_runs")
EVAL_PY = os.path.join(_REPO, "experiments", "835_stage4_eval", "eval_vector.py")
IMPRINT_PY = os.path.join(_REPO, "experiments", "832_linear_imprint", "imprint.py")
FIXTURE_DB = os.path.join(_REPO, "fixtures", "trading_fixture.db")
COLLECTION = os.path.join(EXP_DIR, "chosen_vectors_fixture.json")
CKPT_SRC = os.path.join(EXP_DIR, "ckpt_teacher_src")
CANON_835 = os.path.join(_REPO, "experiments", "835_stage4_eval", "results_fixture_smoke.json")

# The fixture panel both modules must agree on. bar_minutes 240 over the fixture's minute slice
# gives T=111 rows, warm/window_size 60 -> 50 decisions; the 819 book warms in 31 rows, so the
# imprint's start_row = max(31, 60) = 60 == the eval's warm (the collection's decision rows).
PANEL = ["--start", "2025-09-14", "--end", "2025-12-12", "--bar-minutes", "240",
         "--n-assets", "8"]
# FIX 4 (fix pass): the collection walks at the TEACHER OF RECORD's position_size_pct (0.1,
# runs/832_e5/results_e5inv854_db_pw0_832_e5__ckpt_imprint_full.json) while the imprint keeps
# its own 0.5/n sizing (runs/832_e5/jobs_imprint_full.manifest) — the exact mismatch that made
# the two pre-registered real legs non-co-runnable. The field is inert under weight-rebalance
# (measured: 1.000 chosen-vector agreement, identical net@21 / final_pv), so the gate exempts it
# CONDITIONALLY and the smoke reproduces the real pairing instead of papering over it.
COLLECT_POSITION_SIZE_PCT = 0.1      # teacher of record
IMPRINT_POSITION_SIZE_PCT = 0.5 / 8  # imprint.py's fixture default (0.5/n_assets)
CANON_832_SMOKE = os.path.join(_REPO, "experiments", "832_linear_imprint",
                               "results_fixture_smoke.json")

# ---- TASK_867: tracked-baseline hygiene (port of the shipped TASK_864 promote gate) ---------
# The failure mode this closes (confirmed module-wide in runs 21/23): BOTH tracked baselines of
# this module used to be rewritten in place by ordinary verification runs — a bare suite run
# clobbered chosen_vectors_fixture.json THREE times in one session with a stamp-only
# created_utc diff, and `--quick` silently DOWNGRADED results_fixture_smoke.json to falsification
# seed 0 where the committed baseline holds seeds 0..3, while still reading complete and green.
# Fix (mirroring imprint.py TASK_864): default invocations write UNTRACKED *_local outputs;
# overwriting a tracked baseline requires the explicit --promote-baselines flag, which is
# recipe-pinned, validity-gated, atomic and dirty-refusing, and prints a per-field diff.
RESULTS_BASELINE = os.path.join(EXP_DIR, "results_fixture_smoke.json")
RESULTS_LOCAL = os.path.join(EXP_DIR, "results_fixture_smoke_local.json")
COLLECTION_LOCAL = os.path.join(EXP_DIR, "chosen_vectors_fixture_local.json")
CANON_PROMOTE_RECIPE = ("python experiments/861_search_teacher_imprint/smoke_e2e.py "
                        "--promote-baselines")
# The promote flag itself is the delta; every OTHER argparse dest (quick, seeds) is compared
# against the canon full-seed recipe re-derived from this module's own argparse defaults.
CANON_PROMOTE_EXEMPT_FIELDS = ("promote_baselines",)
# Collection provenance fields that legitimately move between identical reproductions: the
# wall-clock stamp (the thrice-observed clobber), and the clone-path-dependent db paths (the
# 866 WEIGHT_ON_ORACLE_VOLATILE precedent). Every other leaf is pinned exactly.
COLLECTION_VOLATILE = ("created_utc", "db", "db_abspath")
# Results-report fields excluded from the committed-vs-fresh pin: wall-clock timings, the
# self-referential pin leg itself, and the derived verdict (recomputed fresh every run — a
# tampered committed verdict cannot change this run's own pass/fail).
RESULTS_PIN_EXCLUDED_TOP = ("verdict", "all_passed", "elapsed_seconds")
RESULTS_PIN_SELF_LEG = "vii_baseline_pins"
# The report's invocation/panel identity block, pinned as part of the results comparison.
PANEL_DOC = {"db": "fixtures/trading_fixture.db", "window": ["2025-09-14", "2025-12-12"],
             "bar_minutes": 240, "n_assets": 8, "n_decisions": 50,
             "collect_position_size_pct": COLLECT_POSITION_SIZE_PCT,
             "imprint_position_size_pct": IMPRINT_POSITION_SIZE_PCT}
# FIX 5 (fix pass): the falsification floor moved 0.10 -> 0.05 after an n=10 re-measurement.
FLOOR_NOTE = {
    "floor": 0.05,
    "was": 0.10,
    "why": "n=10 seeds (fixture panel, 80 iters): legit min +0.1248 / max +0.3618 / mean "
           "+0.2649 / sd 0.0717 vs shuffled min -0.2265 / max -0.0799 / mean -0.1611 / sd "
           "0.0428. Cloud gap 0.2047 (Cohen's d 7.2) — the separation is excellent, the "
           "placement was not: at 0.10 the legit margin is only 0.0248 (0.35 sd, ~1% false-FAIL "
           "per seed, ~4% over a 4-seed sweep). 0.05 leaves 1.04 sd on the legit side and "
           "3.04 sd on the sabotage side, and still sits above the 0.0 a constant marginal "
           "predictor can never beat.",
}

# The +8.024 teacher operator of record (TASK_861 "PIN EXACTLY"), verbatim from
# runs/832_e5/results_e5inv854_db_pw0_832_e5__ckpt_imprint_full.json. Fixture-scale panel, the
# SAME operator block — the smoke exercises the pinned operator, not a cheaper stand-in.
TEACHER_OPERATOR = [
    "--operator", "search", "--vector-proposal-scheme", "dithered-book",
    "--vector-k", "16", "--sims", "32", "--vector-hold-bias", "3.0",
    "--vector-dither-scale", "1.0", "--vector-proposal-temp", "1.0",
    "--vector-proposal-explore-slots", "2", "--prior-weight", "0.0",
    "--leaf-eval", "true-env", "--leaf-rollout-r", "3", "--no-leaf-vhat-tail",
    "--gumbel-scale", "3.0", "--vector-weight-rebalance",
]

# Acceptance (i), 832 half: the scored values of the DEFAULT (flag-OFF) fixture smoke, captured
# from the tree BEFORE this task's diff (`imprint.py` with no new flags). Float-exact.
CANON_832_PINS = {
    "policy_net21": -6.777552797287864,
    "floor_net21": -3.4366469676609452,
    "floor_gross": -1.8107111093013257,
    "teacher_action_agreement": 0.8725,
    "policy_ce_first": 2.5999953746795654,
    "policy_ce_last": 0.299086332321167,
    "gate_rel_diff": 0.09477011466699295,
    "final_pv_net21": 95567.6015625,
}
# Keys of the 835 scorecard that legitimately move with the invocation (out-dir pointer) or the
# calendar (the LEDGER row stamps date.today()) — excluded from the canon-pin comparison.
CANON_835_VOLATILE = ("out_name", "out_pointer", "ledger_row")

# ---- TASK_866: weight-ON parity pin (the 864 incident class, closed) -------------------------
# ONE scored-field mapping shared by the weight-OFF pins (leg i) and the weight-ON parity pin
# (leg i-b): the two configurations are pinned by the SAME fields, each against an oracle of
# its OWN configuration.
PIN_FIELDS_832 = {
    "policy_net21": ("sanity", "per_tier", "21", "policy_inenv"),
    "floor_net21": ("teacher", "floor_inenv", "21"),
    "floor_gross": ("teacher", "floor_inenv", "0"),
    "teacher_action_agreement": ("sanity", "teacher_action_agreement"),
    "policy_ce_first": ("imprint", "losses", "policy_loss", "first"),
    "policy_ce_last": ("imprint", "losses", "policy_loss", "last"),
    "gate_rel_diff": ("expressibility_gate", "rel_diff"),
    "final_pv_net21": ("sanity", "per_tier", "21", "final_pv"),
}
# Top-level keys of the 832 artifact that legitimately move with the invocation: both record
# ABSOLUTE paths (SPEC.md "Reproducibility precondition"), so they are clone-path dependent and
# --ckpt-out dependent. Every other key is a scored/deterministic product of the recipe and is
# compared float-exact.
WEIGHT_ON_ORACLE_VOLATILE = ("db", "checkpoint")
# The configuration the tracked baseline MUST be (SPEC.md "Committed fixture artifact" /
# "Promoting recipe"): weight-ON, default order-code mapping. Recorded by the run itself under
# /imprint — the 864 promote gate refuses any other recipe onto that path.
WEIGHT_ON_RECIPE_IDENTITY = {"vector_weight_rebalance": True, "imprint_mapping": "order-code"}
WEIGHT_OFF_RECIPE_IDENTITY = {"vector_weight_rebalance": False, "imprint_mapping": "order-code"}


def _pin_fields_832(doc):
    """Extract the PIN_FIELDS_832 scored values from an 832 artifact (KeyError = loud schema
    break, which is itself a pin failure)."""
    out = {}
    for name, path in PIN_FIELDS_832.items():
        node = doc
        for key in path:
            node = node[key]
        out[name] = node
    return out


def _recipe_identity_832(doc):
    """The configuration that PRODUCED an 832 artifact, as the artifact itself records it."""
    im = doc.get("imprint") or {}
    cfg = im.get("config") or {}
    return {"vector_weight_rebalance": cfg.get("vector_weight_rebalance"),
            "imprint_mapping": im.get("imprint_mapping")}


def _load_832_weight_on_oracle():
    """Read the tracked 832 baseline AS the weight-ON oracle — FAIL-CLOSED.

    A missing/deleted oracle, an oracle whose git-cleanliness cannot be verified, a git-DIRTY
    oracle (the pinned bytes must be the COMMITTED ones, mirroring
    imprint.py::_assert_baseline_promotable), a corrupt oracle, or an oracle that is not itself
    the weight-ON order-code configuration all REFUSE loud — none of them may degrade into a
    skipped or vacuously-passing pin."""
    rel = os.path.relpath(CANON_832_SMOKE, _REPO)
    if not os.path.exists(CANON_832_SMOKE):
        raise SystemExit(
            f"[866] REFUSING the weight-ON parity pin: the oracle {rel} is MISSING. The pin "
            f"reads the tracked baseline as its oracle, so a deleted oracle must refuse loud, "
            f"never fail open. Restore it (git checkout -- {rel}) and re-run.")
    try:
        p = subprocess.run(["git", "-C", _REPO, "status", "--porcelain", "--", CANON_832_SMOKE],
                           capture_output=True, text=True, timeout=120)
        status = p.stdout if p.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        status = None
    if status is None:
        raise SystemExit(
            f"[866] REFUSING the weight-ON parity pin: could not read `git status` for {rel} "
            f"(git missing, or {_REPO} is not a work tree) — the oracle is only pinnable from "
            f"a verifiably clean working copy (the imprint.py TASK_864 promote-gate rule).")
    if status.strip():
        raise SystemExit(
            f"[866] REFUSING the weight-ON parity pin: the oracle {rel} is git-DIRTY\n"
            f"          {status.strip()}\n"
            f"  The pin guards the COMMITTED bytes. Commit the refresh (with the SPEC.md "
            f"'Promoting recipe' record updated, per TASK_864) or `git checkout -- {rel}`, "
            f"then re-run — pinning against uncommitted edits would bless an unreviewed "
            f"baseline, the exact 864 laundering path.")
    try:
        with open(CANON_832_SMOKE) as fh:
            oracle = json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(
            f"[866] REFUSING the weight-ON parity pin: the oracle {rel} is unreadable/corrupt "
            f"({e}). A truncated or hand-edited oracle must refuse loud, never fail open.")
    identity = _recipe_identity_832(oracle)
    if identity != WEIGHT_ON_RECIPE_IDENTITY:
        raise SystemExit(
            f"[866] REFUSING the weight-ON parity pin: the oracle {rel} is NOT the weight-ON "
            f"order-code configuration it is pinned as — "
            + "; ".join(f"{f}: oracle records {identity.get(f)!r}, expected "
                        f"{WEIGHT_ON_RECIPE_IDENTITY[f]!r}" for f in WEIGHT_ON_RECIPE_IDENTITY)
            + ". A wrong-configuration oracle is the 864 incident itself; refuse, never "
              "compare. Regenerate with the canon recipe (imprint.py --vector-weight-rebalance "
              "--write-parity-baseline).")
    return oracle


def check_832_parity_against_oracle(run_doc, oracle_doc, run_label, oracle_label):
    """Pin a fresh 832 run against an oracle artifact. RECIPE IDENTITY FIRST: a
    different-configuration artifact vs oracle is a REFUSAL with the mismatch NAMED (never a
    numeric diff — the 864 misdiagnosis happened precisely because the only available oracle
    was a different configuration). Same configuration => float-exact comparison of every key
    except WEIGHT_ON_ORACLE_VOLATILE. Returns (ok, differing_top_level_keys)."""
    r_id, o_id = _recipe_identity_832(run_doc), _recipe_identity_832(oracle_doc)
    mism = [f for f in sorted(o_id) if r_id.get(f) != o_id[f]]
    if mism:
        detail = "; ".join(f"{f}: {run_label} is {r_id.get(f)!r} but the oracle "
                           f"({oracle_label}) is {o_id[f]!r}" for f in mism)
        raise SystemExit(
            f"[866] CROSS-CONFIGURATION REFUSAL: {run_label} and the oracle ({oracle_label}) "
            f"are DIFFERENT configurations — {detail}. Comparing them would be the "
            f"wrong-oracle bug this pin exists to close; check the artifact against an oracle "
            f"of its OWN configuration.")
    strip = lambda d: {k: v for k, v in d.items() if k not in WEIGHT_ON_ORACLE_VOLATILE}
    o, r = strip(oracle_doc), strip(run_doc)
    diff = sorted(k for k in set(o) | set(r) if canon_json(o.get(k)) != canon_json(r.get(k)))
    return not diff, diff


def _expect_config_refusal(run_doc, oracle_doc, run_label, oracle_label, field):
    """Demonstration harness: the cross-configuration refusal MUST fire and MUST name the
    mismatching field. Returns (ok, message)."""
    try:
        check_832_parity_against_oracle(run_doc, oracle_doc, run_label, oracle_label)
    except SystemExit as e:
        msg = str(e)
        return ("CROSS-CONFIGURATION REFUSAL" in msg and field in msg), msg
    return False, "no refusal raised — the pin compared two DIFFERENT configurations"


# TASK_865 leg (vi): the SAME ``PANEL`` (minute source, 240-min stride, 8 slots) with the universe
# PINNED. Six of the fixture's minute symbols have coverage in the window and two (`/EURGBP`,
# `/USDCAD`) have none — interleaved at slots 1 and 4, so a slot shift cannot coincidentally line
# up and the inactive-slot disclosure has something to disclose.
MINUTE_FRAME_SYMBOLS = ["XBT/USD", "/EURGBP", "ETH/USD", "LTC/USD",
                        "/USDCAD", "XMR/USD", "ZEC/USD", "ETC/USD"]
MINUTE_FRAME_INACTIVE_SLOTS = [1, 4]
# Slots 5 and 7 list one bar late => PARTIAL (pre-listing rows flat-zero, force-HOLD). `ZEC/USD` at
# slot 6 is a STALE TAIL (minute data ends 2025-09-30; the last panel BAR carrying a real print is
# row 103 of 110): under D30 (ruled 2026-07-30, opt (i)) its 7 trailing rows are FORWARD-FILLED at
# that close and stay TRADEABLE at a frozen price, so it contributes ZERO force-HOLD cells and is
# NOT partial — it is disclosed on its own axis (`stale_tail_slots` / `frozen_slots`), the only
# place it appears. Force-HOLD cells are therefore 2*50 = 100 (it was 107 while the tail was zeroed,
# a treatment MEASURED to fabricate -/+25% of PV; see the (0) PV-conservation leg of the 865 smoke).
# Exhaustive assertions live in the 865 smoke.
MINUTE_FRAME_PARTIAL_SLOTS = [5, 7]
MINUTE_FRAME_STALE_TAIL_SLOTS = [6]
MINUTE_FRAME_FROZEN_SLOTS = [6]
MINUTE_FRAME_FORCE_HOLD_CELLS = 100
# FIX PASS — the honest CELL-level partition of this leg's 50x8 decision grid, OBSERVED out of the
# artifact: 100 INERT (no price at all => force-HOLD) + 6 FROZEN (forward-filled, TRADEABLE: the
# stale tail's 6 in-decision rows) + 294 LIVE. Asserted here so the 861 chain's own coverage of the
# path its real legs run on cannot report a grid as fully live when it is not.
MINUTE_FRAME_CELL_HONESTY = {"n_decision_cells": 400, "n_inert_decision_cells": 100,
                             "n_frozen_decision_cells": 6,
                             "n_frozen_interior_decision_cells": 0,
                             "n_frozen_stale_tail_decision_cells": 6,
                             "n_live_decision_cells": 294}
# D30 rider (b): 6 of 8 slots are covered by design here, below the shared default breadth floor —
# opt out EXPLICITLY rather than weakening the default that guards the real legs.
MINUTE_FRAME_REQUIRE_ACTIVE = ["--require-active-slots", "0"]


def sh(cmd, log, expect_ok=True, quiet_ok=True):
    """Run a subprocess, tee its output to ``log``, return (returncode, text)."""
    t0 = time.time()
    print(f"  $ {' '.join(str(c) for c in cmd[1:])[:150]}")
    p = subprocess.run([str(c) for c in cmd], cwd=_REPO, capture_output=True, text=True)
    text = p.stdout + p.stderr
    with open(log, "w") as fh:
        fh.write(text)
    print(f"    -> exit {p.returncode} in {time.time() - t0:.1f}s  (log {os.path.basename(log)})")
    if expect_ok and p.returncode != 0:
        print(text[-3000:])
        raise SystemExit(f"FATAL: expected success but got exit {p.returncode}: {log}")
    if not quiet_ok:
        print(text[-2000:])
    return p.returncode, text


def canon_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


# ---- TASK_867 helpers (ported from imprint.py's TASK_864 block, same semantics) --------------

def _json_leaves(obj, prefix=""):
    """Flatten a JSON document to (path, scalar) pairs -- the per-field unit of the diff."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _json_leaves(v, f"{prefix}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _json_leaves(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def _diff_leaf_paths(old, new):
    """Sorted leaf paths whose scalar differs (or exists on only one side)."""
    o, n = dict(_json_leaves(old)), dict(_json_leaves(new))
    absent = object()
    return sorted(k for k in set(o) | set(n) if o.get(k, absent) != n.get(k, absent))


def _git_status_porcelain(path):
    """`git status --porcelain` for one path; None when cleanliness cannot be VERIFIED."""
    try:
        p = subprocess.run(["git", "-C", _REPO, "status", "--porcelain", "--", path],
                           capture_output=True, text=True, timeout=120)
        return p.stdout if p.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _assert_one_baseline_clean(path, label):
    """Fail-CLOSED: a missing, git-unverifiable or git-DIRTY tracked baseline refuses loud
    (mirroring imprint.py::_assert_baseline_promotable and the 866 oracle loader)."""
    rel = os.path.relpath(path, _REPO)
    if not os.path.exists(path):
        raise SystemExit(
            f"[867] REFUSING: the tracked {label} baseline {rel} is MISSING. The pins read the "
            f"committed bytes, so a deleted baseline must refuse loud, never fail open. "
            f"Restore it (git checkout -- {rel}) and re-run.")
    status = _git_status_porcelain(path)
    if status is None:
        raise SystemExit(
            f"[867] REFUSING: could not read `git status` for the tracked {label} baseline "
            f"{rel} (git missing, or {_REPO} is not a work tree) — the baselines are only "
            f"pinnable from a verifiably clean working copy (the TASK_864 rule).")
    if status.strip():
        raise SystemExit(
            f"[867] REFUSING: the tracked {label} baseline {rel} is git-DIRTY\n"
            f"          {status.strip()}\n"
            f"  The pins guard the COMMITTED bytes. Commit the refresh or "
            f"`git checkout -- {rel}`, then re-run — running against uncommitted edits would "
            f"bless an unreviewed baseline, the exact 864 laundering path.")


def _assert_tracked_baselines_clean():
    """Pre-run hygiene gate on BOTH tracked baselines, on EVERY invocation (promote or not):
    cheap, and it is what makes a hand-edit of a committed baseline an immediate loud FAIL."""
    _assert_one_baseline_clean(COLLECTION, "collection")
    _assert_one_baseline_clean(RESULTS_BASELINE, "results report")


def _assert_promote_recipe(args, canon):
    """--promote-baselines pins the RECIPE, not just the destination (864 FIX 1). The canon
    recipe is the FULL-seed default: expected values are re-derived from this module's own
    argparse defaults (canon = ap.parse_args([])), never a second hardcoded copy. A seeds-subset
    run (--quick or --seeds N<4) promoted onto the seeds-0..3 baseline is the silent downgrade
    this gate exists to kill, and it REFUSES here, naming every offending field."""
    bad = [(f, getattr(args, f, "<missing>"), want)
           for f, want in sorted(vars(canon).items())
           if f not in CANON_PROMOTE_EXEMPT_FIELDS and getattr(args, f, "<missing>") != want]
    if bad:
        raise SystemExit(
            "[867] --promote-baselines REFUSED: this is not the CANON full-seed recipe, so "
            "promoting it would install a DIFFERENT artifact as the tracked oracle (a "
            "seeds-subset report over the seeds-0..3 baseline is the --quick downgrade, "
            "TASK_867 scope 2).\n"
            + "".join(f"    --{f.replace('_', '-')}: got {g!r}, canon expects {w!r}\n"
                      for f, g, w in bad)
            + f"  The canon recipe is EXACTLY: {CANON_PROMOTE_RECIPE}\n"
              f"  (expected values re-derived from this module's argparse defaults, not a "
              f"second hardcoded copy).")


def _assert_collection_valid(doc):
    """Validity gate on the collection artifact being promoted (864 FIX 2 analogue): schema
    identity, the payload digest recomputed from the decisions block, and the decision-count
    self-consistency. Independent of any other flag or check."""
    bad = []
    if doc.get("schema") != "861/chosen-vectors/v1":
        bad.append(f"schema = {doc.get('schema')!r} (need '861/chosen-vectors/v1')")
    prov = doc.get("provenance") or {}
    dec = doc.get("decisions") or {}
    if not isinstance(dec.get("chosen_vectors"), list) or not dec["chosen_vectors"]:
        bad.append("decisions.chosen_vectors is missing/empty (no recorded stream)")
    else:
        if payload_digest(dec) != prov.get("payload_sha256"):
            bad.append("payload_sha256 does NOT match the decisions block (stale or tampered "
                       "digest — the artifact failed its own integrity seal)")
        n = prov.get("n_decisions_collected")
        counts = {len(dec["chosen_vectors"])}
        for k in ("row_index", "timestamp"):
            if isinstance(dec.get(k), list):
                counts.add(len(dec[k]))
        if n is None or len(counts) != 1 or n not in counts:
            bad.append(f"n_decisions_collected = {n!r} but the payload lengths are "
                       f"{sorted(counts)} (internally inconsistent artifact)")
    if bad:
        raise SystemExit(
            "[867] --promote-baselines REFUSED: the collection artifact did NOT pass its own "
            "validity gate, so it must not become the tracked oracle (TASK_864 FIX 2 "
            "analogue).\n"
            + "".join(f"    - {b}\n" for b in bad)
            + "  The tracked baseline is UNCHANGED. Fix the run, then re-run the canon "
              f"recipe: {CANON_PROMOTE_RECIPE}")


def _assert_report_promotable(report, verdict, canon_seeds):
    """Validity gate on the results report being promoted (864 FIX 2 analogue): every leg must
    have passed EXCEPT the parity-vs-old pin leg (a pin failure is parity drift against the OLD
    baseline — precisely what a reviewed promotion installs — not an internal-invalidity), and
    the recorded seed set must be the canon full-seed identity."""
    bad = []
    if list(report.get("seeds") or []) != list(canon_seeds):
        bad.append(f"seeds = {report.get('seeds')!r}, the canon full-seed identity is "
                   f"{list(canon_seeds)!r} (a seeds-subset report is the --quick downgrade)")
    for k, v in sorted(verdict.items()):
        if k == RESULTS_PIN_SELF_LEG:
            continue
        if not v:
            bad.append(f"leg {k} did not pass — an artifact that was never green must not "
                       f"become the tracked oracle")
    if bad:
        raise SystemExit(
            "[867] --promote-baselines REFUSED: the results report did NOT pass its own "
            "validity gate, so it must not become the tracked oracle (TASK_864 FIX 2 "
            "analogue).\n"
            + "".join(f"    - {b}\n" for b in bad)
            + "  The tracked baseline is UNCHANGED. Fix the run, then re-run the canon "
              f"recipe: {CANON_PROMOTE_RECIPE}")


def _expect_refusal(fn, token):
    """Demonstration harness: the refusal MUST fire (SystemExit) and MUST name ``token``.
    Returns (ok, message)."""
    try:
        fn()
    except SystemExit as e:
        msg = str(e)
        return token in msg, msg
    return False, "no refusal raised"


def _promote_baseline(new_doc, dst_path, label, dump_kwargs, max_show=40):
    """Print the per-field diff vs the checked-in baseline, then replace it ATOMICALLY.

    Port of imprint.py::promote_parity_baseline: the git-dirty gate is re-checked HERE (an edit
    landing during the ~15min run is not silently lost), the promoted bytes are serialised from
    THIS run's in-memory document with the producer's json params (never re-read from the shared
    local path), and the write goes to <baseline>.tmp + os.replace so an interrupt can never
    leave the tracked oracle truncated."""
    _assert_one_baseline_clean(dst_path, label)
    rel = os.path.relpath(dst_path, _REPO)
    new_text = json.dumps(new_doc, **dump_kwargs)
    print("\n" + "=" * 84)
    print(f"[867] --promote-baselines: per-field diff vs the checked-in {rel}")
    print(f"  promoting recipe: {CANON_PROMOTE_RECIPE}")
    print(f"  bytes source: this run's in-memory document")
    with open(dst_path) as fh:
        old = dict(_json_leaves(json.load(fh)))
    new = dict(_json_leaves(json.loads(new_text)))
    moved = sorted(k for k in set(old) & set(new) if old[k] != new[k])
    removed, added = sorted(set(old) - set(new)), sorted(set(new) - set(old))
    print(f"  leaves {len(old)} -> {len(new)}   moved {len(moved)}   "
          f"removed {len(removed)}   added {len(added)}")
    for sym, keys, fmt in (
            ("~", moved, lambda k: f"{old[k]!r}  ->  {new[k]!r}"),
            ("-", removed, lambda k: f"was {old[k]!r}"),
            ("+", added, lambda k: f"= {new[k]!r}")):
        for k in keys[:max_show]:
            print(f"    {sym} {k}\n        {fmt(k)}")
        if len(keys) > max_show:
            print(f"    ... {len(keys) - max_show} more '{sym}' field(s)")
    if not (moved or removed or added):
        print("  IDENTICAL -- the baseline already matches this run")
    tmp = dst_path + ".tmp"
    try:
        with open(tmp, "w") as fh:
            fh.write(new_text)
        os.replace(tmp, dst_path)                        # atomic; no truncated oracle window
    finally:
        if os.path.exists(tmp):                          # only reachable if the write failed
            os.unlink(tmp)
    print(f"[867] {label} baseline PROMOTED -> {rel}  (md5 "
          f"{hashlib.md5(new_text.encode()).hexdigest()})")
    print("      A refresh commit MUST carry the per-field attribution table (every moved "
          "field traced to the commit that moved it) -- TASK_864 acceptance (iii).")
    print("=" * 84)


def mint_teacher_checkpoint():
    """Fresh-init checkpoint for the collection walk. The pinned teacher operator is pw0 +
    true-env leaf + no v̂ tail (net_dependent=false — LEDGER 832/E5-exit2), so its decisions do
    not read the weights at all; the checkpoint exists only to satisfy the harness' load path."""
    if os.path.exists(os.path.join(CKPT_SRC, "trainer_metadata.json")):
        print(f"[861-smoke] reusing fresh-init teacher checkpoint {CKPT_SRC}")
        return
    import importlib.util
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)
    spec = importlib.util.spec_from_file_location("eval_vector861", EVAL_PY)
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    ns = argparse.Namespace(
        model_size="small", h_cycles=4, l_cycles=6, sims=32, discount=0.997, gumbel_scale=3.0,
        value_scale=1.0, value_shift=0.0, vector_k=16, vector_hold_bias=3.0,
        vector_dither_scale=1.0, vector_proposal_scheme="dithered-book",
        vector_proposal_temp=1.0, vector_proposal_explore_slots=2, vector_budget_step=False,
        vector_budget_pct=0.99, vector_weight_rebalance=True, vector_weight_unit=-1.0,
        vector_weight_max=4.0, vector_rebalance_deadband=-1.0)
    _model, trainer = ev.build_model_and_trainer(ns, 8, init_seed=0)
    os.makedirs(CKPT_SRC, exist_ok=True)
    trainer.save_checkpoint(os.path.abspath(CKPT_SRC))
    print(f"[861-smoke] minted fresh-init teacher checkpoint -> {CKPT_SRC}")


def eval_cmd(out_dir, collect=None, collect_raw=None):
    cmd = [sys.executable, EVAL_PY, "--db", FIXTURE_DB, *PANEL,
           "--position-size-pct", str(COLLECT_POSITION_SIZE_PCT), *TEACHER_OPERATOR,
           "--checkpoint", CKPT_SRC, "--out-dir", out_dir, "--tag", "861fx"]
    if collect is not None:
        cmd += ["--collect-chosen-vectors", collect]
    if collect_raw is not None:                       # empty-string / bad-path legs
        cmd += ["--collect-chosen-vectors", collect_raw]
    return cmd


def imprint_cmd(out_json, ckpt_out, seed=0, collection=COLLECTION, shuffle=False, extra=()):
    # --allow-off-pin-teacher is REQUIRED here and nowhere else: the smoke collects under a
    # checkpoint it just minted, which can never equal the pinned runs/832_e5/ckpt_imprint_full.
    # Leg iii below pins the honesty of that escape hatch — the ONLY off-pin field may be
    # 'checkpoint'; every operator knob must still match the +8.024 pin.
    cmd = [sys.executable, IMPRINT_PY, *PANEL, "--vector-weight-rebalance",
           "--teacher-vectors", collection, "--allow-off-pin-teacher",
           "--iters", "80", "--seed", str(seed),
           "--ckpt-out", ckpt_out, "--out", out_json]
    if shuffle:
        cmd.append("--shuffle-teacher-vectors")
    return cmd + list(extra)


def payload_digest(decisions):
    """Mirror of eval_vector.payload_digest — so a tamper leg can recompute a VALID digest and
    prove the deeper checks (integrality, replay score) stand on their own."""
    blob = json.dumps({k: v for k, v in decisions.items() if k != "note"},
                      sort_keys=True, separators=(",", ":"), default=float)
    return hashlib.sha256(blob.encode()).hexdigest()


def tamper(path, mutate, reseal):
    """Write a mutated copy of the legit collection. ``reseal`` recomputes the payload digest
    (so the leg tests the check BEYOND the digest); otherwise the digest is left stale."""
    doc = copy.deepcopy(json.load(open(COLLECTION)))
    mutate(doc)
    if reseal:
        doc["provenance"]["payload_sha256"] = payload_digest(doc["decisions"])
    with open(path, "w") as fh:
        json.dump(doc, fh)
    return path


def main():
    ap = argparse.ArgumentParser(description="TASK_861 end-to-end fixture smoke")
    ap.add_argument("--quick", action="store_true",
                    help="one seed per falsification arm instead of four")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--promote-baselines", dest="promote_baselines",
                    action="store_true", default=False,
                    help="TASK_867: promote this run's outputs onto the TWO CHECKED-IN "
                         "baselines (chosen_vectors_fixture.json + results_fixture_smoke.json). "
                         "Default OFF => the run writes the gitignored *_local.json files, so "
                         "NO ordinary verification run can dirty (or, under --quick, silently "
                         "DOWNGRADE) a tracked baseline. Recipe-pinned to the canon full-seed "
                         "invocation, validity-gated, atomic (.tmp + os.replace), "
                         "dirty-refusing, per-field diff printed. Canon recipe: "
                         + CANON_PROMOTE_RECIPE)
    args = ap.parse_args()
    # TASK_867: the canon promoting recipe, re-derived from this module's own argparse defaults
    # (no second hardcoded copy of any default).
    canon_args = ap.parse_args([])
    canon_seeds = list(range(canon_args.seeds))
    # TASK_867: fail-CLOSED hygiene gate BEFORE anything expensive — a missing or git-dirty
    # tracked baseline refuses every invocation loud (a hand-edited baseline is an immediate
    # FAIL, never a silent read). The promote recipe pin is likewise pre-run: a refused
    # promotion costs nothing.
    _assert_tracked_baselines_clean()
    if args.promote_baselines:
        _assert_promote_recipe(args, canon_args)
    seeds = list(range(1 if args.quick else args.seeds))
    os.makedirs(RUN_DIR, exist_ok=True)
    report = {"legs": {}, "seeds": seeds}
    t_start = time.time()

    # ---------------------------------------------------------------- (i) canon pins
    print("\n=== (i) CANON PINS — both flags OFF must be bit-identical ===")
    canon_dir = os.path.join(RUN_DIR, "canon835")
    os.makedirs(canon_dir, exist_ok=True)
    # The smoke checkpoint goes to a leg-local dir (never the shared default) so a concurrent
    # run in experiments/835_stage4_eval/ cannot collide with it; the two provenance fields that
    # then carry that path are normalized back to the committed artifact's value before the
    # comparison, so every SCORED key is still pinned byte-for-byte.
    canon_ck = os.path.join(RUN_DIR, "canon835_ckpt")
    sh([sys.executable, EVAL_PY, "--smoke", "--out-dir", canon_dir,
        "--smoke-ckpt-dir", canon_ck], os.path.join(RUN_DIR, "canon_835_smoke.log"))
    got = json.load(open(os.path.join(canon_dir, "results_fixture_smoke.json")))
    want = json.load(open(CANON_835))
    got["checkpoint"] = want["checkpoint"]
    if isinstance(got.get("roundtrip"), dict) and isinstance(want.get("roundtrip"), dict):
        got["roundtrip"]["ckpt_dir"] = want["roundtrip"]["ckpt_dir"]
    strip = lambda d: {k: v for k, v in d.items() if k not in CANON_835_VOLATILE}
    ok_835 = canon_json(strip(got)) == canon_json(strip(want))
    print(f"  835 fixture smoke vs the checked-in artifact "
          f"(minus {CANON_835_VOLATILE}): {'IDENTICAL' if ok_835 else 'DIFFERS'}")
    if not ok_835:
        diff = [k for k in strip(want) if canon_json(want.get(k)) != canon_json(got.get(k))]
        print(f"  differing keys: {diff}")
    report["legs"]["i_canon_835"] = {
        "passed": bool(ok_835),
        "baseline": "committed experiments/835_stage4_eval/results_fixture_smoke.json",
        "excluded_keys": list(CANON_835_VOLATILE) + ["checkpoint", "roundtrip.ckpt_dir"],
        "excluded_reason": "invocation-dependent (out-dir / smoke-ckpt-dir paths) or "
                           "calendar-dependent (the LEDGER row stamps date.today()); every "
                           "SCORED key is compared byte-for-byte"}

    canon832 = os.path.join(RUN_DIR, "canon_832_default.json")
    sh([sys.executable, IMPRINT_PY, "--out", canon832,
        "--ckpt-out", os.path.join(RUN_DIR, "canon832_ckpt")],
       os.path.join(RUN_DIR, "canon_832_smoke.log"))
    c = json.load(open(canon832))
    got832 = _pin_fields_832(c)
    # TASK_866: the weight-OFF pins are only a meaningful oracle for a weight-OFF artifact —
    # verify the configuration the run RECORDED before comparing numbers (a weight-ON artifact
    # against these pins must be a named config mismatch, not a numeric diff).
    id832 = _recipe_identity_832(c)
    id832_ok = id832 == WEIGHT_OFF_RECIPE_IDENTITY
    bad = {k: (v, got832[k]) for k, v in CANON_832_PINS.items() if got832[k] != v}
    ok_832 = not bad and id832_ok and "teacher_vectors" not in c
    print(f"  832 default smoke vs the pre-change pins: "
          f"{'IDENTICAL' if ok_832 else 'DIFFERS ' + str(bad)}")
    print(f"  832 default recipe identity {id832} -> "
          f"{'weight-OFF order-code as pinned' if id832_ok else 'CONFIG MISMATCH — REFUSED'}")
    report["legs"]["i_canon_832"] = {"passed": bool(ok_832), "pins": CANON_832_PINS,
                                     "observed": got832,
                                     "recipe_identity": id832,
                                     "recipe_identity_is_weight_off_order_code": bool(id832_ok),
                                     "additive_block_absent": "teacher_vectors" not in c}

    # ---------------- (i-b) TASK_866: the WEIGHT-ON configuration is pinned against the tracked
    # baseline READ AS ITS OWN-CONFIGURATION ORACLE (the 861 pins above are weight-OFF and could
    # never see a weight-ON move — the 864 incident class).
    print("\n=== (i-b) WEIGHT-ON PARITY PIN (TASK_866) — oracle of the SAME configuration ===")
    won_out = os.path.join(RUN_DIR, "canon_832_weight_on.json")
    sh([sys.executable, IMPRINT_PY, "--vector-weight-rebalance", "--out", won_out,
        "--ckpt-out", os.path.join(RUN_DIR, "canon832_won_ckpt")],
       os.path.join(RUN_DIR, "canon_832_weight_on.log"))
    w = json.load(open(won_out))
    oracle = _load_832_weight_on_oracle()        # fail-CLOSED: missing/dirty/wrong-config refuses
    ok_won, diff_won = check_832_parity_against_oracle(
        w, oracle, "fresh --vector-weight-rebalance run", "tracked baseline (weight-ON)")
    print(f"  832 weight-ON canon run vs the tracked baseline "
          f"(minus {WEIGHT_ON_ORACLE_VOLATILE}): "
          f"{'IDENTICAL' if ok_won else 'DIFFERS ' + str(diff_won)}")

    # Acceptance (i), weight-ON direction: perturb ONE weight-ON field in a COPY of the oracle
    # (the tracked file is never touched) -> the pin MUST fail, naming the owning key.
    mut = copy.deepcopy(oracle)
    mut["teacher"]["floor_inenv"]["21"] += 1e-3
    ok_mut, diff_mut = check_832_parity_against_oracle(
        w, mut, "fresh --vector-weight-rebalance run", "PERTURBED baseline copy")
    ok_mut_demo = (not ok_mut) and "teacher" in diff_mut
    print(f"  mutation (weight-ON): teacher.floor_inenv.21 +1e-3 in the oracle copy -> pin "
          f"{'FAILS as required' if ok_mut_demo else 'STILL PASSES — PIN DEAD'} "
          f"(differing keys {diff_mut}); unmutated check above is the restore->green half")

    # Acceptance (i), weight-OFF direction stays live: perturb ONE scored field of the flagless
    # artifact -> the CANON_832_PINS comparison MUST fail.
    c_mut = copy.deepcopy(c)
    c_mut["sanity"]["per_tier"]["21"]["policy_inenv"] += 1e-3
    got_mut = _pin_fields_832(c_mut)
    bad_mut_off = {k: (v, got_mut[k]) for k, v in CANON_832_PINS.items() if got_mut[k] != v}
    ok_off_demo = list(bad_mut_off) == ["policy_net21"]
    print(f"  mutation (weight-OFF): sanity.per_tier.21.policy_inenv +1e-3 -> weight-OFF pins "
          f"{'FAIL as required' if ok_off_demo else 'DID NOT FAIL CLEANLY — PIN DEAD'} "
          f"(fields {sorted(bad_mut_off)}); the live leg-(i) check above is the restore->green half")

    # Acceptance (ii): cross-configuration refusal, BOTH directions, mismatch NAMED.
    ok_ref_off, msg_off = _expect_config_refusal(
        c, oracle, "flagless weight-OFF run", "tracked weight-ON baseline",
        "vector_weight_rebalance")
    print(f"  weight-OFF artifact vs weight-ON oracle -> "
          f"{'REFUSED, mismatch named' if ok_ref_off else 'NOT REFUSED'}")
    print(f"    {msg_off.splitlines()[0] if msg_off else ''}")
    ok_ref_on, msg_on = _expect_config_refusal(
        w, c, "fresh weight-ON run", "weight-OFF default run used as oracle",
        "vector_weight_rebalance")
    print(f"  weight-ON artifact vs weight-OFF oracle -> "
          f"{'REFUSED, mismatch named' if ok_ref_on else 'NOT REFUSED'}")
    print(f"    {msg_on.splitlines()[0] if msg_on else ''}")

    ib_checks = {
        "weight_on_run_matches_tracked_baseline": bool(ok_won),
        "mutation_weight_on_field_fails_pin": bool(ok_mut_demo),
        "mutation_weight_off_field_fails_pins": bool(ok_off_demo),
        "weight_off_artifact_vs_weight_on_oracle_REFUSED": bool(ok_ref_off),
        "weight_on_artifact_vs_weight_off_oracle_REFUSED": bool(ok_ref_on),
    }
    for k, v in ib_checks.items():
        print(f"  {'OK  ' if v else 'FAIL'} {k}")
    report["legs"]["ib_weight_on_pin"] = {
        "passed": all(ib_checks.values()), "checks": ib_checks,
        "oracle": "experiments/832_linear_imprint/results_fixture_smoke.json (tracked, "
                  "weight-ON order-code; fail-closed on missing/dirty/wrong-config)",
        "oracle_recipe_identity": WEIGHT_ON_RECIPE_IDENTITY,
        "excluded_keys": list(WEIGHT_ON_ORACLE_VOLATILE),
        "excluded_reason": "absolute invocation paths (clone-path / --ckpt-out dependent, "
                           "SPEC.md 'Reproducibility precondition'); every scored key is "
                           "compared float-exact",
        "observed": _pin_fields_832(w),
        "differing_keys": diff_won,
        "weight_off_refusal_message": msg_off,
        "weight_on_refusal_message": msg_on}

    # ------------------------------------------------- (ii) collection is a pure recorder
    print("\n=== (ii) COLLECTION IS A PURE RECORDER — flag ON vs OFF, byte-identical ===")
    mint_teacher_checkpoint()
    off_dir = os.path.join(RUN_DIR, "walk_off")
    on_dir = os.path.join(RUN_DIR, "walk_on")
    for d in (off_dir, on_dir):
        os.makedirs(d, exist_ok=True)
    sh(eval_cmd(off_dir), os.path.join(RUN_DIR, "walk_off.log"))
    # TASK_867: the collection walk writes the UNTRACKED *_local path — the tracked
    # chosen_vectors_fixture.json is only ever moved by --promote-baselines (leg vii pins the
    # fresh local reproduction against the committed bytes).
    sh(eval_cmd(on_dir, collect=COLLECTION_LOCAL), os.path.join(RUN_DIR, "walk_on.log"))
    name = "results_861fx_861_search_teacher_imprint__ckpt_teacher_src.json"
    a = open(os.path.join(off_dir, name), "rb").read()
    b = open(os.path.join(on_dir, name), "rb").read()
    # out_dir is the ONLY thing that differs between the two invocations, and it lands in
    # out_pointer/ledger_row — normalize it away, then demand byte equality.
    a_n, b_n = a.replace(off_dir.encode(), b"OUTDIR"), b.replace(on_dir.encode(), b"OUTDIR")
    a_n = a_n.replace(os.path.relpath(off_dir, _REPO).encode(), b"OUTDIR")
    b_n = b_n.replace(os.path.relpath(on_dir, _REPO).encode(), b"OUTDIR")
    ok_rec = a_n == b_n
    print(f"  scorecard JSON with vs without --collect-chosen-vectors: "
          f"{'BYTE-IDENTICAL' if ok_rec else 'DIFFERS'}")
    coll = json.load(open(COLLECTION))
    cp = coll["provenance"]
    print(f"  collection: {cp['n_decisions_collected']} decisions x {cp['n_assets']} slots, "
          f"rows {cp['start_row']}.., net@{cp['cost_tier_bp']}, schema {coll['schema']}")
    walk_net21 = json.load(open(os.path.join(on_dir, name)))["tiers"]["21"]["sharpe_full"]
    report["legs"]["ii_pure_recorder"] = {
        "passed": bool(ok_rec), "n_decisions": cp["n_decisions_collected"],
        "collection_walk_net21": walk_net21, "schema": coll["schema"]}

    # ------------------------------------------------------ (iii) provenance fail-loud
    print("\n=== (iii) PROVENANCE FAIL-LOUD — mismatched window / frame / env must be refused ===")
    def _rev_syms(d):
        d["provenance"]["loaded_symbols"] = list(reversed(d["provenance"]["loaded_symbols"]))

    def _off_pin_sims(d):
        d["provenance"]["operator"]["sims"] = 8

    def _byte_edit(d):
        d["decisions"]["chosen_vectors"][0][0] = (
            0 if d["decisions"]["chosen_vectors"][0][0] != 0 else 2)

    def _non_integer(d):
        d["decisions"]["chosen_vectors"] = [[v - 0.5 for v in row]
                                            for row in d["decisions"]["chosen_vectors"]]

    def _permute_payload(d):
        cv = d["decisions"]["chosen_vectors"]
        d["decisions"]["chosen_vectors"] = cv[len(cv) // 2:] + cv[:len(cv) // 2]

    def _order_env_psp(d):
        # FIX 4 negative branch: with weight-rebalance OFF on BOTH sides the position_size_pct
        # exemption is NOT applicable and the same mismatch must be a hard refusal.
        d["provenance"]["env_flags"]["vector_weight_rebalance"] = False
        d["provenance"]["env_flags"]["position_size_pct"] = 0.1

    tampered = tamper(os.path.join(RUN_DIR, "tampered_frame_collection.json"),
                      _rev_syms, reseal=False)
    tam_offpin = tamper(os.path.join(RUN_DIR, "tampered_offpin_sims8.json"),
                        _off_pin_sims, reseal=False)
    tam_bytes = tamper(os.path.join(RUN_DIR, "tampered_byte_edit.json"),
                       _byte_edit, reseal=False)        # stale digest -> caught by sha256
    tam_float = tamper(os.path.join(RUN_DIR, "tampered_non_integer.json"),
                       _non_integer, reseal=True)       # VALID digest -> caught by integrality
    tam_perm = tamper(os.path.join(RUN_DIR, "tampered_permuted.json"),
                      _permute_payload, reseal=True)    # VALID digest, integral, in-range,
                                                        # on-pin -> only the walk score catches it
    tam_psp = tamper(os.path.join(RUN_DIR, "tampered_order_env_psp.json"),
                     _order_env_psp, reseal=False)
    refuse_ck = os.path.join(RUN_DIR, "refuse_ckpt")
    def _imp(out, coll, extra=(), weight=True, offpin=True):
        cmd = [sys.executable, IMPRINT_PY, *PANEL]
        if weight:
            cmd.append("--vector-weight-rebalance")
        cmd += ["--teacher-vectors", coll]
        if offpin:
            cmd.append("--allow-off-pin-teacher")
        return cmd + ["--iters", "5", "--ckpt-out", refuse_ck, "--out", out] + list(extra)

    def _out(name):
        return os.path.join(RUN_DIR, f"refuse_{name}.json")

    refusals = [
        # window: the same collection, an imprint panel that starts one day later.
        ("window", "window.start", "PROVENANCE REFUSAL", _out("window"),
         [sys.executable, IMPRINT_PY, "--start", "2025-09-15", "--end", "2025-12-12",
          "--bar-minutes", "240", "--n-assets", "8", "--vector-weight-rebalance",
          "--teacher-vectors", COLLECTION, "--allow-off-pin-teacher", "--iters", "5",
          "--ckpt-out", refuse_ck, "--out", _out("window")]),
        # frame: the recorded universe order is reversed -> slot k would be a different asset.
        ("frame", "loaded_symbols", "PROVENANCE REFUSAL", _out("frame"),
         _imp(_out("frame"), tampered)),
        # env: the collection walked the weight-rebalance env, the imprint would walk orders.
        ("env_flags", "env_flags.vector_weight_rebalance", "PROVENANCE REFUSAL", _out("env"),
         _imp(_out("env"), COLLECTION, weight=False)),
        # FIX 4 negative branch: position_size_pct is exempt ONLY under weight semantics.
        ("env_psp_conditional", "env_flags.position_size_pct", "PROVENANCE REFUSAL",
         _out("env_psp"), _imp(_out("env_psp"), tam_psp, weight=False)),
        # FIX 1: an off-pin operator (sims 8, not the pinned 32) must fail loud WITHOUT the
        # escape hatch — the falsification gate cannot see this (it scored a -7.08 garbage
        # teacher HIGHER than the pinned +8.02 one).
        ("off_pin_operator", "operator.sims", "OFF-PIN TEACHER REFUSAL", _out("offpin"),
         _imp(_out("offpin"), tam_offpin, offpin=False)),
        # FIX 2: a byte-edited payload (stale digest).
        ("payload_digest", "PAYLOAD DIGEST MISMATCH", "PAYLOAD DIGEST MISMATCH",
         _out("digest"), _imp(_out("digest"), tam_bytes)),
        # FIX 2c: every code rewritten as x-0.5, digest RESEALED -> shape and [0,4] range still
        # pass; only the integrality check stands between this and a -0.649 teacher.
        ("payload_non_integer", "NON-INTEGER action code", "PROVENANCE REFUSAL",
         _out("nonint"), _imp(_out("nonint"), tam_float)),
        # FIX 2b: payload PERMUTED and resealed — passes provenance, the operator pin, the
        # digest and integrality. Only the recorded walk score catches it.
        ("replay_integrity", "REPLAY INTEGRITY FAILURE", "REPLAY INTEGRITY FAILURE",
         _out("replay"), _imp(_out("replay"), tam_perm)),
    ]
    prov = {}
    for label, token, keyword, artifact, cmd in refusals:
        if os.path.exists(artifact):
            os.remove(artifact)                       # never judge a stale artifact
        rc, txt = sh(cmd, os.path.join(RUN_DIR, f"refuse_{label}.log"), expect_ok=False)
        named, refused_kw = token in txt, keyword in txt
        wrote = os.path.exists(artifact)
        ok = rc != 0 and named and refused_kw and not wrote
        print(f"  {label:<20} exit {rc} | '{keyword}' {refused_kw} | "
              f"names '{token}' {named} | wrote artifact {wrote} -> "
              f"{'REFUSED' if ok else 'NOT REFUSED'}")
        prov[label] = {"passed": bool(ok), "exit_code": rc, "named_field": token,
                       "keyword": keyword, "field_named": bool(named),
                       "artifact_written": bool(wrote)}
    report["legs"]["iii_provenance"] = prov

    # extra fail-loud: one collection path cannot serve several checkpoints
    rc, txt = sh([sys.executable, EVAL_PY, "--db", FIXTURE_DB, *PANEL,
                  "--position-size-pct", str(COLLECT_POSITION_SIZE_PCT), *TEACHER_OPERATOR,
                  "--checkpoint", CKPT_SRC, "--checkpoint", CKPT_SRC,
                  "--out-dir", off_dir, "--collect-chosen-vectors",
                  os.path.join(RUN_DIR, "never_written.json")],
                 os.path.join(RUN_DIR, "refuse_multickpt.log"), expect_ok=False)
    ok_multi = rc != 0 and "single walk" in txt
    print(f"  multi-ckpt exit {rc} -> {'REFUSED' if ok_multi else 'NOT REFUSED'}")
    report["legs"]["iii_provenance"]["multi_checkpoint"] = {"passed": bool(ok_multi),
                                                            "exit_code": rc}

    # ------------------------------------------- (iii-b) FIX 3: empty/blank flags fail CLOSED
    print("\n=== (iii-b) EMPTY-STRING FLAGS MUST FAIL CLOSED (FIX 3) ===")
    blank = {}
    canon_before = open(CANON_832_SMOKE, "rb").read()
    # eval side: --collect-chosen-vectors "" used to run the WHOLE walk and exit 0 writing
    # nothing. It must now abort at argparse time, i.e. before the panel is even built.
    t0 = time.time()
    rc, txt = sh(eval_cmd(off_dir, collect_raw=""),
                 os.path.join(RUN_DIR, "blank_collect.log"), expect_ok=False)
    dt = time.time() - t0
    ok_blank_eval = rc != 0 and "EMPTY path" in txt and "[835] evaluating" not in txt
    print(f"  eval --collect-chosen-vectors '' : exit {rc} in {dt:.1f}s, no walk started "
          f"({'[835] evaluating' not in txt}) -> {'REFUSED' if ok_blank_eval else 'FAIL-OPEN'}")
    blank["eval_empty_collect_path"] = {"passed": bool(ok_blank_eval), "exit_code": rc,
                                        "seconds": round(dt, 2),
                                        "walk_started": "[835] evaluating" in txt}
    # eval side: an unwritable output path must be rejected BEFORE the walk, not after it.
    # parent is an existing regular FILE -> genuinely unwritable, discovered at preflight
    bad_path = os.path.join(COLLECTION, "nested.json")
    t0 = time.time()
    rc, txt = sh(eval_cmd(off_dir, collect_raw=bad_path),
                 os.path.join(RUN_DIR, "unwritable_collect.log"), expect_ok=False)
    dt = time.time() - t0
    ok_unwritable = rc != 0 and "not writable" in txt and "[835] evaluating" not in txt
    print(f"  eval --collect-chosen-vectors <unwritable> : exit {rc} in {dt:.1f}s, no walk "
          f"started -> {'REFUSED' if ok_unwritable else 'FAIL-OPEN'}")
    blank["eval_unwritable_collect_path"] = {"passed": bool(ok_unwritable), "exit_code": rc,
                                             "seconds": round(dt, 2),
                                             "walk_started": "[835] evaluating" in txt}
    # imprint side: --teacher-vectors "" used to train the DEFAULT 819 teacher AND overwrite the
    # checked-in canon parity baseline (tv_suffix stayed empty).
    rc, txt = sh([sys.executable, IMPRINT_PY, *PANEL, "--vector-weight-rebalance",
                  "--teacher-vectors", "", "--iters", "5"],
                 os.path.join(RUN_DIR, "blank_teacher_vectors.log"), expect_ok=False)
    canon_after = open(CANON_832_SMOKE, "rb").read()
    ok_blank_imp = rc != 0 and "EMPTY path" in txt and canon_after == canon_before
    print(f"  imprint --teacher-vectors '' : exit {rc}, canon baseline byte-identical "
          f"{canon_after == canon_before} -> "
          f"{'REFUSED' if ok_blank_imp else 'FAIL-OPEN'}")
    blank["imprint_empty_teacher_vectors"] = {
        "passed": bool(ok_blank_imp), "exit_code": rc,
        "canon_832_baseline_untouched": canon_after == canon_before,
        "baseline": "experiments/832_linear_imprint/results_fixture_smoke.json"}
    report["legs"]["iiib_empty_flags"] = blank

    # ---------------------------------------------- (iv) falsification, seed-swept
    print("\n=== (iv) FALSIFICATION — legit passes / shuffled fails, on every seed ===")
    arms = {"legit": [], "shuffled": []}
    for arm in ("legit", "shuffled"):
        for s in seeds:
            out = os.path.join(RUN_DIR, f"imprint_{arm}_s{s}.json")
            rc, txt = sh(imprint_cmd(out, os.path.join(RUN_DIR, f"ck_{arm}_s{s}"), seed=s,
                                     shuffle=(arm == "shuffled")),
                         os.path.join(RUN_DIR, f"imprint_{arm}_s{s}.log"), expect_ok=False)
            fit = None
            if os.path.exists(out):
                fit = json.load(open(out))["teacher_vectors"]["fit_vs_original_targets"]
            red = None if fit is None else fit["nll_reduction"]
            want_ok = (arm == "legit")
            ok = (rc == 0) if want_ok else (rc != 0)
            print(f"  {arm:<9} seed {s}: exit {rc}  NLL-reduction "
                  f"{'n/a' if red is None else f'{red:+.4f}'} -> {'PASS' if ok else 'FAIL'}")
            arms[arm].append({"seed": s, "exit_code": rc, "nll_reduction": red,
                              "expected_exit_zero": want_ok, "passed": bool(ok),
                              "argmax_agreement_vs_original":
                                  None if fit is None else fit["argmax_agreement_vs_original"],
                              "modal_action_rate":
                                  None if fit is None else fit["modal_action_rate"]})
    report["legs"]["iv_falsification"] = arms

    # ------------------------------- (iv-b) escape-hatch HONESTY PIN + margin against the floor
    legit0 = os.path.join(RUN_DIR, f"imprint_legit_s{seeds[0]}.json")
    r0 = json.load(open(legit0))
    tv0 = r0["teacher_vectors"]
    off_fields = tv0["operator_off_pin_fields"]
    ok_honest = off_fields == ["checkpoint"]
    print("\n=== (iv-b) ESCAPE-HATCH HONESTY — the smoke may be off-pin in ONE field only ===")
    print(f"  operator_off_pin={tv0['operator_off_pin']} fields={off_fields} -> "
          f"{'HONEST (checkpoint only)' if ok_honest else 'TOO PERMISSIVE'}")
    exempt_notes = tv0["source"]["gate"]["env_exempt_mismatches"]
    psp_exempted = any(n.startswith("position_size_pct:") for n in exempt_notes)
    print(f"  conditional env exemption actually exercised (position_size_pct "
          f"{COLLECT_POSITION_SIZE_PCT} collection vs {IMPRINT_POSITION_SIZE_PCT} imprint): "
          f"{psp_exempted}")
    ri = tv0["replay_integrity"]
    print(f"  in-runner replay integrity: checked={ri['checked']} rel={ri.get('rel_delta')}")
    floors = [a["nll_reduction"] for a in arms["legit"]]
    ceils = [a["nll_reduction"] for a in arms["shuffled"] if a["nll_reduction"] is not None]
    report["legs"]["ivb_escape_hatch_honesty"] = {
        "passed": bool(ok_honest and psp_exempted and ri["checked"]),
        "operator_off_pin": tv0["operator_off_pin"],
        "operator_off_pin_fields": off_fields,
        "position_size_pct_exemption_exercised": bool(psp_exempted),
        "env_exempt_mismatches": exempt_notes,
        "in_runner_replay_integrity": ri,
        "floor": FLOOR_NOTE,
        "legit_nll_reduction_min": (min(floors) if floors else None),
        "shuffled_nll_reduction_max": (max(ceils) if ceils else None)}

    # ------------------------ (v) the two PRE-REGISTERED real legs must be CO-RUNNABLE verbatim
    print("\n=== (v) PINNED-RECIPE CO-RUNNABILITY — leg 1 collection + leg 2 recipe together ===")
    vout = os.path.join(RUN_DIR, "imprint_pinned_recipe.json")
    rc, txt = sh(imprint_cmd(vout, os.path.join(RUN_DIR, "ck_pinned_recipe"), seed=0,
                             extra=("--imprint-mapping", "weight-native")),
                 os.path.join(RUN_DIR, "imprint_pinned_recipe.log"), expect_ok=False)
    rv = json.load(open(vout)) if os.path.exists(vout) else {}
    ok_v = rc == 0 and rv.get("imprint", {}).get("imprint_mapping") == "search-teacher-vectors"
    print(f"  --imprint-mapping weight-native + --teacher-vectors (position_size_pct "
          f"{COLLECT_POSITION_SIZE_PCT} vs {IMPRINT_POSITION_SIZE_PCT}): exit {rc} -> "
          f"{'CO-RUNNABLE' if ok_v else 'BLOCKED'}")
    report["legs"]["v_pinned_recipe_corunnable"] = {
        "passed": bool(ok_v), "exit_code": rc,
        "imprint_mapping": rv.get("imprint", {}).get("imprint_mapping"),
        "note": "leg 1 argv (teacher of record, position_size_pct 0.1) and leg 2 argv "
                "(jobs_imprint_full.manifest recipe: weight-native, 0.5/n sizing) run together; "
                "the 845 --vector-weight-rebalance fail-closed assert is still enforced on this "
                "path"}

    # ------------------------------- (vi) TASK_865: the same chain on the MINUTE FRAME PIN
    print("\n=== (vi) MINUTE FRAME PIN — collect + imprint with the universe PINNED ===")

    def _write_frame(path, symbols):
        with open(path, "w") as fh:
            json.dump({"frame_symbols": list(symbols), "n_assets": len(symbols),
                       "train_window": ["2025-09-14", "2025-12-12"],
                       "db": "fixtures/trading_fixture.db",
                       "source": "861 smoke leg (vi) — TASK_865 minute frame pin"}, fh, indent=1)
        return path

    mf = _write_frame(os.path.join(RUN_DIR, "minute_frame.json"), MINUTE_FRAME_SYMBOLS)
    mf_reordered = list(MINUTE_FRAME_SYMBOLS)
    mf_reordered[0], mf_reordered[2] = mf_reordered[2], mf_reordered[0]
    mf_bad = _write_frame(os.path.join(RUN_DIR, "minute_frame_reordered.json"), mf_reordered)
    m_dir = os.path.join(RUN_DIR, "walk_minute_pin")
    os.makedirs(m_dir, exist_ok=True)
    m_coll = os.path.join(RUN_DIR, "chosen_vectors_minute_pin.json")
    m_pin = ["--panel-frame-from-checkpoint", mf, "--panel-frame-source", "minute",
             *MINUTE_FRAME_REQUIRE_ACTIVE]
    sh(eval_cmd(m_dir, collect=m_coll) + m_pin, os.path.join(RUN_DIR, "walk_minute_pin.log"))
    m_out = os.path.join(RUN_DIR, "imprint_minute_pin.json")
    rc_m, _ = sh(imprint_cmd(m_out, os.path.join(RUN_DIR, "ck_minute_pin"), seed=0,
                             collection=m_coll, extra=tuple(m_pin)),
                 os.path.join(RUN_DIR, "imprint_minute_pin.log"), expect_ok=False)
    rm = json.load(open(m_out)) if os.path.exists(m_out) else {}
    mpf = rm.get("panel_frame") or {}
    # ... and the positional-honesty direction: same membership, different ORDER -> REFUSED.
    bad_out = os.path.join(RUN_DIR, "imprint_minute_pin_reordered.json")
    if os.path.exists(bad_out):
        os.remove(bad_out)
    rc_bad, txt_bad = sh(imprint_cmd(bad_out, os.path.join(RUN_DIR, "ck_minute_pin_bad"), seed=0,
                                     collection=m_coll,
                                     extra=("--panel-frame-from-checkpoint", mf_bad,
                                            "--panel-frame-source", "minute",
                                            *MINUTE_FRAME_REQUIRE_ACTIVE, "--iters", "5")),
                         os.path.join(RUN_DIR, "imprint_minute_pin_reordered.log"),
                         expect_ok=False)
    m_checks = {
        "imprint_exit_zero": rc_m == 0,
        "panel_frame_source_minute": (rm.get("panel_frame_source") == "minute"
                                      and mpf.get("panel_frame_source") == "minute"),
        "frame_symbols_in_recorded_order": rm.get("frame_symbols") == MINUTE_FRAME_SYMBOLS,
        "inactive_slots_disclosed": rm.get("inactive_slots") == MINUTE_FRAME_INACTIVE_SLOTS,
        "partial_slots_head_only": rm.get("partial_slots") == MINUTE_FRAME_PARTIAL_SLOTS,
        "stale_tail_disclosed_on_its_own_axis": (
            rm.get("stale_tail_slots") == MINUTE_FRAME_STALE_TAIL_SLOTS
            and mpf.get("stale_tail_slots") == MINUTE_FRAME_STALE_TAIL_SLOTS),
        "stale_tail_slot_is_in_NEITHER_other_list": all(
            sl not in (rm.get("inactive_slots") or [])
            and sl not in (rm.get("partial_slots") or [])
            for sl in MINUTE_FRAME_STALE_TAIL_SLOTS),
        "force_hold_cells_exact_stale_tail_TRADEABLE": (
            mpf.get("n_force_hold_cells") == MINUTE_FRAME_FORCE_HOLD_CELLS),
        # FIX PASS: frozen spans (interior INCLUDED) are disclosed, and the decision grid's honest
        # INERT / FROZEN / LIVE partition is OBSERVED out of the artifact, not assumed.
        "frozen_slots_disclosed": (rm.get("frozen_slots") == MINUTE_FRAME_FROZEN_SLOTS
                                   and mpf.get("frozen_slots") == MINUTE_FRAME_FROZEN_SLOTS),
        "decision_cell_honesty_exact": all(
            (rm.get("decision_cell_honesty") or {}).get(k) == v
            for k, v in MINUTE_FRAME_CELL_HONESTY.items()),
        "breadth_floor_recorded": mpf.get("require_active_slots") == 0.0,
        "sanity_replay_both_tiers": bool(rm) and all(
            rm["sanity"]["per_tier"][t]["policy_inenv"] is not None for t in ("0", "21")),
        "replay_integrity_passed": bool(
            (rm.get("teacher_vectors") or {}).get("replay_integrity", {}).get("passed")),
        "reordered_frame_REFUSED": (rc_bad != 0
                                    and "panel_frame identity MISMATCH" in txt_bad
                                    and not os.path.exists(bad_out)),
    }
    for k, v in m_checks.items():
        print(f"  {'OK  ' if v else 'FAIL'} {k}")
    report["legs"]["vi_minute_frame_pin"] = {
        "passed": all(m_checks.values()), "checks": m_checks,
        "frame": MINUTE_FRAME_SYMBOLS, "inactive_slots": rm.get("inactive_slots"),
        "partial_slots": rm.get("partial_slots"), "stale_tail_slots": rm.get("stale_tail_slots"),
        "frozen_slots": rm.get("frozen_slots"),
        "decision_cell_honesty": rm.get("decision_cell_honesty"),
        "sanity_net21": (rm.get("sanity") or {}).get("per_tier", {}).get("21", {}).get(
            "policy_inenv"),
        "note": "same window/stride/size as the legacy 861 panel — only the universe is PINNED "
                "instead of coverage-ranked (TASK_865). Exhaustive matrix: "
                "experiments/865_minute_frame_pin/smoke_minute_frame_pin.py",
        "sanity_net21_attribution": (
            "FIX PASS — do NOT attribute a swing in this number to the stale tail's tradeability. "
            "The stale-tail treatment changes exactly the tail cells of ONE slot; replaying the "
            "SAME recorded teacher stream on the filled and zeroed panels isolates that channel "
            "and it is SMALL (measured and reported by the (i-b) walk_net21 reconciliation leg of "
            "experiments/865_minute_frame_pin/smoke_minute_frame_pin.py, which also carries the "
            "sign). A before/after difference in sanity_net21 is dominated by the operator "
            "RE-DECIDING on a different panel and by the student being retrained on that different "
            "stream — not by the tail becoming tradeable.")}

    # ------------------------------------------------------- round-trip cross-check
    floor21 = r0["teacher"]["floor_inenv"]["21"]
    delta = None if (floor21 is None or walk_net21 is None) else abs(floor21 - walk_net21)
    # Tolerance, not equality: 835 marks PV through a BATCHED jax.vmap(compute_portfolio_value)
    # while 832's walk_env calls it unbatched, so the float32 reduction order differs. Anything
    # beyond ~1e-4 relative would mean a genuinely different trajectory, not rounding.
    rel = None if delta is None else delta / max(abs(walk_net21), 1.0)
    ok_rt = rel is not None and rel < 1e-4
    print("\n=== ROUND-TRIP: imprint replay of the collected vectors vs the collection walk ===")
    print(f"  imprint teacher floor net@21 {floor21:+.6f} vs collection walk net@21 "
          f"{walk_net21:+.6f} (|d| {delta:.2e}, rel {rel:.2e}) -> "
          f"{'MATCH' if ok_rt else 'MISMATCH'}")
    report["legs"]["roundtrip_teacher_floor"] = {
        "passed": bool(ok_rt), "imprint_floor_net21": floor21,
        "collection_walk_net21": walk_net21, "abs_delta": delta, "rel_delta": rel,
        "rel_tolerance": 1e-4,
        "note": "same actions, same rows, same env — walked by 835 rollout (batched vmap PV) and "
                "832 walk_env (unbatched PV); agreement to float32 rounding proves the recorded "
                "target stream replays as the identical trajectory"}
    report["imprint_sanity_first_seed"] = {
        "policy_net21": r0["sanity"]["per_tier"]["21"]["policy_inenv"],
        "teacher_floor_net21": floor21,
        "corridor_position_net21": r0["ledger_row"]["imprint_corridor_position_net21"],
        "teacher_action_agreement": r0["sanity"]["teacher_action_agreement"],
        "note": "fixture plumbing numbers — NO threshold applied (the binding in-sample bar is "
                "the planner's real-panel leg 3: sanity replay net@21 > +0.839)"}

    # ----------------------- (vii) TASK_867: TRACKED-BASELINE PINS — the COMMITTED bytes must
    # match what the current code + recipe reproduces. The pre-867 checks here were within-run
    # byte-identity only and passed against ARBITRARY committed content — the named defect.
    print("\n=== (vii) TRACKED-BASELINE PINS (TASK_867) — committed baselines vs fresh "
          "reproduction ===")

    def _coll_view(doc):
        d = copy.deepcopy(doc)
        for k in COLLECTION_VOLATILE:
            (d.get("provenance") or {}).pop(k, None)
        return d

    def _results_view(doc):
        d = copy.deepcopy(doc)
        for k in RESULTS_PIN_EXCLUDED_TOP:
            d.pop(k, None)
        legs = d.get("legs") or {}
        legs.pop(RESULTS_PIN_SELF_LEG, None)               # self-referential: records this pin
        for sub in (legs.get("iiib_empty_flags") or {}).values():
            if isinstance(sub, dict):
                sub.pop("seconds", None)                   # wall-clock timing
        return d

    # Collection half: this run's FRESH collection (written to the untracked *_local path in
    # leg ii) vs the COMMITTED baseline, minus the volatile stamp/clone-path fields. Presence
    # and git-cleanliness of the committed file were already gated pre-run (fail-closed).
    fresh_coll = json.load(open(COLLECTION_LOCAL))
    try:
        committed_coll = json.load(open(COLLECTION))
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(
            f"[867] REFUSING the collection pin: the committed baseline "
            f"{os.path.relpath(COLLECTION, _REPO)} is unreadable/corrupt ({e}). A truncated or "
            f"hand-edited baseline must refuse loud, never fail open.")
    coll_diff = _diff_leaf_paths(_coll_view(committed_coll), _coll_view(fresh_coll))
    ok_coll_pin = not coll_diff
    print(f"  collection: fresh reproduction vs the committed baseline (minus "
          f"{COLLECTION_VOLATILE}): "
          f"{'IDENTICAL' if ok_coll_pin else 'DIFFERS ' + str(coll_diff[:8])}")

    # Results half: the FULL-seed recipe reproduces the committed report float-exact (minus the
    # documented volatile/derived fields). A seeds-subset run (--quick) CANNOT reproduce a
    # seeds-0..3 baseline, so it SKIPS the value comparison — SAYING SO (acceptance iv) — and
    # instead validates the committed baseline's recorded recipe/seed-set identity, which is
    # exactly the identity the pre-867 --quick downgrade destroyed.
    try:
        committed_results = json.load(open(RESULTS_BASELINE))
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(
            f"[867] REFUSING the results pin: the committed baseline "
            f"{os.path.relpath(RESULTS_BASELINE, _REPO)} is unreadable/corrupt ({e}). A "
            f"truncated or hand-edited baseline must refuse loud, never fail open.")
    ok_seed_identity = committed_results.get("seeds") == canon_seeds
    full_recipe = (not args.quick) and args.seeds == canon_args.seeds
    fresh_results_view = _results_view({**report, "panel": PANEL_DOC,
                                        "teacher_operator": TEACHER_OPERATOR})
    res_diff = None
    if full_recipe:
        res_diff = _diff_leaf_paths(_results_view(committed_results), fresh_results_view)
        ok_res_pin = not res_diff and ok_seed_identity
        print(f"  results: fresh full-seed report vs the committed baseline (minus "
              f"{RESULTS_PIN_EXCLUDED_TOP}, the self-referential pin leg and iiib timings): "
              f"{'IDENTICAL' if not res_diff else 'DIFFERS ' + str(res_diff[:8])}")
    else:
        ok_res_pin = ok_seed_identity
        print(f"  results: seeds {seeds} are a SUBSET of the canon {canon_seeds} — the value "
              f"comparison is SKIPPED (said out loud per acceptance iv; --quick must never "
              f"stand in for the full-seed baseline). Validating the committed baseline's "
              f"seed-set identity instead: recorded {committed_results.get('seeds')} -> "
              f"{'canon full-seed' if ok_seed_identity else 'NOT the full-seed baseline — REFUSED'}")

    # Scope-2/864 demonstrations, in-leg and cheap: the promote machinery MUST refuse (a) the
    # --quick downgrade, (b) a seeds-subset, (c) an invalid collection artifact (stale payload
    # digest), (d) an invalid report (a red leg) — each naming what it refused.
    ok_demo_quick, msg_demo_quick = _expect_refusal(
        lambda: _assert_promote_recipe(
            argparse.Namespace(quick=True, seeds=canon_args.seeds, promote_baselines=True),
            canon_args), "quick")
    ok_demo_seeds, msg_demo_seeds = _expect_refusal(
        lambda: _assert_promote_recipe(
            argparse.Namespace(quick=False, seeds=1, promote_baselines=True),
            canon_args), "seeds")
    bad_coll = copy.deepcopy(fresh_coll)
    bad_coll["decisions"]["chosen_vectors"][0][0] = (
        0 if bad_coll["decisions"]["chosen_vectors"][0][0] != 0 else 2)   # digest now stale
    ok_demo_coll, msg_demo_coll = _expect_refusal(
        lambda: _assert_collection_valid(bad_coll), "payload")
    ok_demo_rep, msg_demo_rep = _expect_refusal(
        lambda: _assert_report_promotable(report, {"iv_falsification": False}, canon_seeds),
        "iv_falsification")
    print(f"  promote-gate demos: --quick REFUSED {ok_demo_quick} | seeds-subset REFUSED "
          f"{ok_demo_seeds} | invalid collection REFUSED {ok_demo_coll} | invalid report "
          f"REFUSED {ok_demo_rep}")

    vii_checks = {
        "collection_pin_fresh_vs_committed": bool(ok_coll_pin),
        "results_pin": bool(ok_res_pin),
        "committed_results_seed_identity_is_canon_full_seed": bool(ok_seed_identity),
        "promote_refuses_quick_downgrade": bool(ok_demo_quick),
        "promote_refuses_seeds_subset": bool(ok_demo_seeds),
        "promote_refuses_invalid_collection": bool(ok_demo_coll),
        "promote_refuses_invalid_report": bool(ok_demo_rep),
    }
    for k, v in vii_checks.items():
        print(f"  {'OK  ' if v else 'FAIL'} {k}")
    report["legs"][RESULTS_PIN_SELF_LEG] = {
        "passed": all(vii_checks.values()), "checks": vii_checks,
        "collection_baseline": "experiments/861_search_teacher_imprint/"
                               "chosen_vectors_fixture.json (tracked; fail-closed on "
                               "missing/dirty/corrupt pre-run)",
        "results_baseline": "experiments/861_search_teacher_imprint/"
                            "results_fixture_smoke.json (tracked; same gate)",
        "collection_volatile_excluded": list(COLLECTION_VOLATILE),
        "collection_differing_leaves": coll_diff,
        "results_compared_full_recipe": bool(full_recipe),
        "results_differing_leaves": res_diff,
        "results_excluded": ("top-level " + str(list(RESULTS_PIN_EXCLUDED_TOP)) +
                             ", the self-referential pin leg, iiib wall-clock seconds; "
                             "verdict/all_passed are recomputed fresh every run"),
        "promote_demonstrations": {
            "quick": msg_demo_quick, "seeds_subset": msg_demo_seeds,
            "invalid_collection": msg_demo_coll, "invalid_report": msg_demo_rep},
        "note": "default invocations write the gitignored *_local.json outputs; the tracked "
                "baselines move ONLY via --promote-baselines (recipe-pinned full-seed, "
                "validity-gated, atomic, dirty-refusing, per-field diff printed — the TASK_864 "
                "gate ported). The dirty/missing refusal of BOTH baselines is enforced "
                "pre-run on every invocation, so a hand-edited committed baseline fails the "
                "smoke before any leg runs (acceptance iii)."}

    # ------------------------------------------------------------------- verdict
    def leg_ok(x):
        if isinstance(x, dict):
            if "passed" in x:
                return bool(x["passed"])
            return all(leg_ok(v) for v in x.values())
        if isinstance(x, list):
            return all(leg_ok(v) for v in x)
        return True
    verdict = {k: leg_ok(v) for k, v in report["legs"].items()}
    report["verdict"] = verdict
    report["all_passed"] = all(verdict.values())
    report["elapsed_seconds"] = round(time.time() - t_start, 1)
    report["panel"] = PANEL_DOC
    report["teacher_operator"] = TEACHER_OPERATOR
    # TASK_867: the report goes to the UNTRACKED *_local path on every invocation — the tracked
    # results_fixture_smoke.json moves ONLY through the explicit promote below.
    out_path = RESULTS_LOCAL
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    print("\n" + "=" * 84)
    for k, v in verdict.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"  elapsed {report['elapsed_seconds']}s   ->  {os.path.relpath(out_path, _REPO)}")
    print("=" * 84)
    if args.promote_baselines:
        # Validity gates are INDEPENDENT of the parity-vs-old pin leg (864 FIX 2 analogue): a
        # pin failure is drift against the OLD baseline — precisely what this reviewed
        # promotion installs — while any OTHER red leg or an invalid collection refuses.
        _assert_collection_valid(fresh_coll)
        _assert_report_promotable(report, verdict, canon_seeds)
        _promote_baseline(fresh_coll, COLLECTION, "collection",
                          dict(indent=1, default=float))   # eval_vector's write params
        _promote_baseline(report, RESULTS_BASELINE, "results report",
                          dict(indent=2, default=float))   # this module's write params above
    else:
        print("[867] tracked baselines UNTOUCHED (default): the collection is at "
              f"{os.path.relpath(COLLECTION_LOCAL, _REPO)} and the report at "
              f"{os.path.relpath(RESULTS_LOCAL, _REPO)}; promotion is explicit: "
              f"{CANON_PROMOTE_RECIPE}")
    if not report["all_passed"]:
        raise SystemExit("861 SMOKE FAIL — see the legs above")
    print("[861] SMOKE PASS: canon bit-identical, recorder inert, provenance fail-loud, "
          "falsification separates on every seed")


if __name__ == "__main__":
    main()
