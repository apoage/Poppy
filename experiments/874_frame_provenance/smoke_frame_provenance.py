#!/usr/bin/env python3
"""TASK_874 — fixture smoke for ``imprint.py``'s frame-provenance EMITTER.

``imprint.py`` CONSUMES a frame provenance (TASK_863 ``--panel-frame-from-checkpoint``) but
never EMITTED one for the checkpoint it writes — every 862-scorecard run over a fresh imprint
needed a provenance file hand-derived first (the gap the 861 leg's own
``runs/861_ref/frame_provenance_ref_s{0,1,2}.json`` ``source`` note flags). This proves the four
acceptance gates on ``fixtures/trading_fixture.db`` only (CPU, small iters; NEVER
``trading_data.db``):

  (i)   a fixture imprint (DERIVED/unpinned frame branch) emits ``frame_provenance_<ckpt>.json``
        alongside its checkpoint; a 7-window toy scorecard battery grades that checkpoint through
        its OWN emitted provenance, no hand-editing, 7/7 windows OK.
  (ii)  pointing that SAME provenance file at a DIFFERENT checkpoint directory is still REFUSED
        by scorecard.py's existing frame/provenance mismatch gate (byte-identical message) — the
        863/865 gate is read-only here, never re-implemented, never weakened.
  (iii) the PINNED emission branch: a matching pin succeeds; a pin that no longer agrees with a
        FRESH re-read of itself (scrambled between validation and emission) is REFUSED and no
        file is written — exercised IN-PROCESS (``imp.emit_frame_provenance`` directly) because
        a mismatched pin is already caught earlier in the SAME run by TASK_863's own gate
        (``load_framed_panel``), so the only way to reach this emitter's OWN independent check is
        to call it directly with a frame that no longer matches the pin on disk.
  (iv)  scope-3 comparison (its own read-only entry point,
        ``experiments/874_frame_provenance/compare_861_ref_provenance.py``) + existing suites
        green + no committed baseline moved (checked by the caller via git).

FIX PASS (verification panel 2026-08-08) — three more legs, one per reproduced BLOCKER:

  (B1) a run that REFUSES ITSELF leaves NO provenance file, so the checkpoint it wrote is
       UNGRADABLE. The panel's repro verbatim: a degenerate PINNED imprint (863 FIX-0
       refusal, exit 1) used to leave a valid provenance file behind, and scorecard.py then
       graded that refused checkpoint to exit 0 / "1 OK / 0 FAILED". This leg asserts the
       refusal fires, no provenance appears, and the scorecard over that checkpoint REFUSES.
  (B2) the DERIVED branch mirrors env_factory's pad-by-modulo, which duplicates symbols when
       the in-window universe is smaller than --n-assets — a provenance scorecard.py refuses by
       construction. The panel's repro verbatim (--n-assets 20 on the fixture's 8-symbol
       universe): the emitter REFUSES, names the duplicates, and writes nothing.
  (B3) an existing provenance file is never silently clobbered: identical content is a quiet
       no-op, differing content is a REFUSAL that leaves the file untouched, and
       --overwrite-frame-provenance is the loud, explicit opt-in.

FINAL FIX PASS (fail-closed judge 2026-08-08) — two more legs:

  (I1) NO COMMITTED BASELINE MOVES. The emission refusal must fire BEFORE
       promote_parity_baseline's os.replace onto the git-tracked results_fixture_smoke.json, not
       after it. Two sub-legs, because the two halves of the claim are reachable from different
       places:
         I1a  end-to-end subprocess: the canon promoting recipe (--vector-weight-rebalance
              --write-parity-baseline) with a CONFLICTING provenance already sitting at the
              default checkpoint's provenance path -> exit 1, no promotion banner, and
              results_fixture_smoke.json sha256-IDENTICAL before and after. This is the smoke's
              first and only exercise of --write-parity-baseline at all.
         I1b  in-process, the RACE the fix actually closes: run()'s pre-training dry run cannot
              see a provenance file that appears after it, so main() is driven with a stub run()
              that drops a conflicting file and hands over a sink document. The assertion is
              ORDERING — main() must exit 1 with promote_parity_baseline NEVER CALLED (it is
              replaced by a non-mutating spy, so this leg can never move the baseline even when
              it FAILS).
  (I2) --overwrite-frame-provenance leaves its trace IN THE ARTIFACT: the replacement document
       carries provenance_replaced=true plus replaced_provenance_sha256 (the sha256 of the raw
       bytes it replaced), and the stamp does NOT turn a later identical re-run into a refusal.
       Folded into (B3).

Run: ``python experiments/874_frame_provenance/smoke_frame_provenance.py``
"""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(EXP_DIR))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
import fixtures.fixture_identity as fixture_identity          # noqa: E402  (TASK_873)

RUN_DIR = os.path.join(EXP_DIR, "smoke_runs")
IMPRINT_PY = os.path.join(_REPO, "experiments", "832_linear_imprint", "imprint.py")
SCORECARD_PY = os.path.join(_REPO, "experiments", "835_stage4_eval", "scorecard.py")
FIXTURE_DB = os.path.join(_REPO, "fixtures", "trading_fixture.db")
FIXTURE_DB_REL = "fixtures/trading_fixture.db"

# The fixture's canonical top-8 minute-coverage-ranked universe (unpinned default; matches every
# other task's fixture smoke that reaches this same ranking, e.g. 862's own FRAME_SYMBOLS).
FIXTURE_FRAME_SYMBOLS = ["XBT/USD", "ETH/USD", "LTC/USD", "XMR/USD",
                        "ZEC/USD", "ETC/USD", "DASH/USD", "REP/USD"]
FIXTURE_PANEL = ["--start", "2025-09-14", "--end", "2025-12-12", "--bar-minutes", "240",
                 "--n-assets", "8"]
GRADE_WINDOW = ("2025-09-14", "2025-12-12")     # same span, MINUTE-sourced (TASK_865)
GRADE_BAR_MINUTES = 60                          # MINUTE_BAR_MINUTES_RANGE floor (scorecard.py)
GRADE_MAX_DECISIONS = 8                         # plumbing proof, not a measurement

# The 863 fixture pin: 6 slots with full ohlcv_daily coverage in the window and 2 with ZERO,
# interleaved (copied from experiments/863_imprint_frame_pin/smoke_frame_pin.py, not re-derived).
PIN_FRAME_SYMBOLS = ["XBT/USD", "/EURGBP", "ETH/USD", "LTC/USD",
                     "/USDCAD", "XMR/USD", "ZEC/USD", "ETC/USD"]
PIN_PANEL = ["--start", "2025-09-14", "--end", "2025-12-12", "--bar-minutes", "1440",
             "--n-assets", "8"]

# ---- (I1) the CANON promoting recipe's own paths (TASK_864) ----
# The git-TRACKED parity baseline, the one artifact in this task's blast radius that a refusal
# must never be able to move, and the provenance path the canon recipe's default --ckpt-out
# (experiments/832_linear_imprint/ckpt_smoke) maps to.
EXP_832 = os.path.join(_REPO, "experiments", "832_linear_imprint")
PARITY_BASELINE = os.path.join(EXP_832, "results_fixture_smoke.json")
CANON_PROV = os.path.join(EXP_832, "frame_provenance_smoke.json")
CANON_PROMOTE_ARGV = ["--vector-weight-rebalance", "--write-parity-baseline"]


def sha256_file(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def conflicting_provenance_doc(note):
    """A syntactically VALID provenance that differs from what the canon recipe would emit (a
    different train_window), i.e. the case write_frame_provenance must REFUSE rather than
    clobber."""
    return {"frame_symbols": FIXTURE_FRAME_SYMBOLS, "n_assets": 8,
            "source": note, "train_window": ["1999-01-01", "1999-12-31"],
            "db": FIXTURE_DB_REL, "checkpoint": "experiments/832_linear_imprint/ckpt_smoke",
            "panel_frame_source": "minute"}


def sh(cmd, log, expect_ok=True):
    t0 = time.time()
    print(f"  $ {' '.join(str(c) for c in cmd[1:])[:170]}")
    p = subprocess.run([str(c) for c in cmd], cwd=_REPO, capture_output=True, text=True)
    text = p.stdout + p.stderr
    with open(log, "w") as fh:
        fh.write(text)
    print(f"    -> exit {p.returncode} in {time.time() - t0:.1f}s  ({os.path.basename(log)})")
    if expect_ok and p.returncode != 0:
        print(text[-4000:])
        raise SystemExit(f"FATAL: expected success but got exit {p.returncode}: {log}")
    return p.returncode, text


def _imprint_module():
    spec = importlib.util.spec_from_file_location("imprint832_smoke874", IMPRINT_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# =============================================================================================
# (i) derive a fixture checkpoint + its OWN emitted provenance, then a 7-window battery over it
# =============================================================================================

def leg_i_derived_emit_and_battery(R):
    print("\n=== (i) DERIVED (unpinned) fixture imprint -> emitted provenance -> 7/7 battery ===")
    ckpt = R("ckpt_derived")
    out_json = R("results_derived.json")
    cmd = [sys.executable, IMPRINT_PY, "--db", FIXTURE_DB, "--vector-weight-rebalance",
           "--iters", "150", "--seed", "0", "--ckpt-out", ckpt, "--out", out_json]
    sh(cmd, R("log_derived.txt"))
    stem = os.path.basename(ckpt)[len("ckpt_"):]
    prov_path = os.path.join(os.path.dirname(ckpt), f"frame_provenance_{stem}.json")
    prov = json.load(open(prov_path)) if os.path.exists(prov_path) else None
    checks = {
        "provenance_file_written": prov is not None,
        "frame_symbols_is_canonical_8": prov is not None
                                        and prov["frame_symbols"] == FIXTURE_FRAME_SYMBOLS,
        "n_assets_8": prov is not None and prov["n_assets"] == 8,
        "checkpoint_is_the_output_path": prov is not None
                                         and prov["checkpoint"].endswith("ckpt_derived"),
        "db_recorded": prov is not None and "trading_fixture.db" in prov["db"],
        "train_window_matches_argv": prov is not None
                                     and prov["train_window"] == ["2025-09-14", "2025-12-12"],
        "source_says_derived_not_pinned": prov is not None and "DERIVED" in prov["source"],
        "panel_frame_source_minute": prov is not None and prov["panel_frame_source"] == "minute",
    }
    for k, v in checks.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    if not all(checks.values()):
        return {"leg": "i_derived_emit", "passed": False, "checks": checks}

    # ---- 7-window toy battery, NO hand-editing: grades ckpt through prov_path as emitted ----
    def floor_artifact(name):
        path = R(f"floor_{name}.json")
        with open(path, "w") as fh:
            json.dump({"tag": f"toy-{name}", "db": FIXTURE_DB_REL, "n_assets": 8,
                       "window": list(GRADE_WINDOW),
                       "floor": {"0": 0.0, "21": 0.0}, "ceiling": {"0": 10.0, "21": 10.0}},
                      fh, indent=2)
        return path

    windows = []
    for name in ("regime_a", "regime_b", "regime_c", "regime_d"):
        windows.append({"name": name, "kind": "regime", "start": GRADE_WINDOW[0],
                        "end": GRADE_WINDOW[1], "floor_artifact": floor_artifact(name),
                        "max_decisions": GRADE_MAX_DECISIONS})
    windows.append({"name": "oos_toy", "kind": "oos", "start": GRADE_WINDOW[0],
                    "end": GRADE_WINDOW[1], "max_decisions": GRADE_MAX_DECISIONS})
    windows.append({"name": "wf_toy_q1", "kind": "walkforward", "start": GRADE_WINDOW[0],
                    "end": GRADE_WINDOW[1], "max_decisions": GRADE_MAX_DECISIONS})
    windows.append({"name": "wf_toy_q2", "kind": "walkforward", "start": GRADE_WINDOW[0],
                    "end": GRADE_WINDOW[1], "max_decisions": GRADE_MAX_DECISIONS})
    manifest = {
        "schema": "imprint_scorecard_manifest/v1", "name": "874smoke_derived",
        "checkpoint": ckpt, "frame_provenance": prov_path,       # <- the EMITTED file, verbatim
        "panel_frame_source": "minute", "db": FIXTURE_DB_REL, "n_assets": 8,
        "bar_minutes": GRADE_BAR_MINUTES, "require_active_slots": 0, "eval_seed": 0, "model": {},
        "notes": "TASK_874 fixture smoke (i) — 7-window toy battery over an imprint.py-emitted "
                 "provenance, zero manual steps.",
        "windows": windows,
    }
    manifest_path = R("manifest_derived.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    out_dir = R("battery_out")
    sh([sys.executable, SCORECARD_PY, "--manifest", manifest_path, "--out-dir", out_dir],
       R("log_battery.txt"))
    sc = json.load(open(os.path.join(out_dir, "scorecard_874smoke_derived.json")))
    bchecks = {
        "status_OK": sc["status"] == "OK",
        "seven_windows": sc["n_windows"] == 7,
        "seven_ok": sc["n_ok"] == 7,
        "zero_failed": sc["n_failed"] == 0,
        "no_global_errors": sc["global_errors"] == [],
        "frame_provenance_path_is_the_emitted_file": sc["frame_provenance"]["path"] in
            (prov_path, os.path.relpath(prov_path, _REPO)),
    }
    for k, v in bchecks.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    print(f"    n_ok/n_windows = {sc['n_ok']}/{sc['n_windows']}  status={sc['status']}")
    checks.update(bchecks)
    return {"leg": "i_derived_emit_and_battery", "passed": all(checks.values()), "checks": checks,
           "ckpt": ckpt, "prov_path": prov_path, "manifest": manifest_path}


# =============================================================================================
# (ii) the SAME provenance graded against a DIFFERENT checkpoint -> still REFUSED
# =============================================================================================

def leg_ii_cross_checkpoint_refused(R, prov_path):
    print("\n=== (ii) cross-checkpoint grading through the SAME provenance -> still REFUSED ===")
    other_ckpt = R("ckpt_other_not_the_one_in_the_provenance")
    os.makedirs(other_ckpt, exist_ok=True)          # preflight only needs os.path.isdir
    manifest = {
        "schema": "imprint_scorecard_manifest/v1", "name": "874smoke_cross_ckpt",
        "checkpoint": other_ckpt, "frame_provenance": prov_path,
        "panel_frame_source": "minute", "db": FIXTURE_DB_REL, "n_assets": 8,
        "bar_minutes": GRADE_BAR_MINUTES, "require_active_slots": 0, "eval_seed": 0, "model": {},
        "notes": "TASK_874 fixture smoke (ii) — cross-checkpoint refusal, gate not weakened.",
        "windows": [{"name": "regime_x", "kind": "regime", "start": GRADE_WINDOW[0],
                    "end": GRADE_WINDOW[1], "floor_artifact": R("floor_regime_a.json"),
                    "max_decisions": GRADE_MAX_DECISIONS}],
    }
    manifest_path = R("manifest_cross_ckpt.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    out_dir = R("battery_out_cross_ckpt")
    rc, text = sh([sys.executable, SCORECARD_PY, "--manifest", manifest_path, "--out-dir", out_dir],
                 R("log_cross_ckpt.txt"), expect_ok=False)
    sc_path = os.path.join(out_dir, "scorecard_874smoke_cross_ckpt.json")
    sc = json.load(open(sc_path)) if os.path.exists(sc_path) else {}
    needle = "was recorded for checkpoint"
    checks = {
        "exit_non_zero": rc != 0,
        "status_FAILED": sc.get("status") == "FAILED",
        "global_error_present": bool(sc.get("global_errors")),
        "message_is_the_863_865_gate_message": any(needle in e for e in sc.get("global_errors", [])),
        "scrambles_asset_identity_wording_intact": any(
            "scrambles asset identity" in e for e in sc.get("global_errors", [])),
        "no_window_ran": all(w.get("status") == "FAILED" and "sharpe" not in
                             (w.get("tiers") or {}).get("21", {}) for w in sc.get("windows", [])),
    }
    for k, v in checks.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    if sc.get("global_errors"):
        print(f"    message: {sc['global_errors'][0][:300]}")
    return {"leg": "ii_cross_checkpoint_refused", "passed": all(checks.values()), "checks": checks}


# =============================================================================================
# (iii) the PINNED emission branch, in-process: a match writes, a scrambled-after-validation
# pin refuses to write (the emitter's OWN independent re-assertion, scope 2)
# =============================================================================================

def leg_iii_pinned_emission_matrix(R, imp):
    print("\n=== (iii) pinned emission: matching pin WRITES; scrambled pin REFUSES to emit ===")
    from types import SimpleNamespace
    pin_path = R("pin_frame.json")
    with open(pin_path, "w") as fh:
        json.dump({"frame_symbols": FIXTURE_FRAME_SYMBOLS, "n_assets": 8,
                   "source": "TASK_874 smoke pin", "train_window": ["2021-01-01", "2025-09-01"],
                   "db": "trading_data.db", "checkpoint": "runs/somewhere/ckpt_upstream"},
                  fh, indent=1)
    args = SimpleNamespace(panel_frame_from_checkpoint=pin_path, panel_frame_source="minute",
                           n_assets=8, start="2025-09-14", end="2025-12-12", db=FIXTURE_DB_REL)

    # ---- positive control: the frame this run "walked" agrees with the pin -> WRITES ----
    ckpt_ok = R("ckpt_pin_match")
    os.makedirs(ckpt_ok, exist_ok=True)
    frame_match = {"frame_symbols": list(FIXTURE_FRAME_SYMBOLS)}
    written = imp.emit_frame_provenance(ckpt_ok, args, frame_match, None)
    doc = json.load(open(written))
    positive_checks = {
        "file_written": os.path.exists(written),
        "frame_symbols_recorded": doc["frame_symbols"] == FIXTURE_FRAME_SYMBOLS,
        "source_names_the_pin": os.path.basename(pin_path) in doc["source"]
                                or "pinned to" in doc["source"],
        "panel_frame_source_minute": doc["panel_frame_source"] == "minute",
    }
    for k, v in positive_checks.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")

    # ---- negative control: the walked frame no longer agrees with a FRESH read of the pin ----
    # (simulates the pin file being scrambled between the run's OWN earlier validation and this
    # emitter's independent re-check — the scenario scope 2 names).
    ckpt_bad = R("ckpt_pin_scrambled")
    os.makedirs(ckpt_bad, exist_ok=True)
    scrambled = list(FIXTURE_FRAME_SYMBOLS)
    scrambled[0], scrambled[1] = scrambled[1], scrambled[0]
    frame_scrambled = {"frame_symbols": scrambled}
    target_path = os.path.join(ckpt_bad, "..",
                               f"frame_provenance_{os.path.basename(ckpt_bad)[len('ckpt_'):]}.json")
    target_path = os.path.normpath(target_path)
    pre_existed = os.path.exists(target_path)
    raised = None
    try:
        imp.emit_frame_provenance(ckpt_bad, args, frame_scrambled, None)
    except SystemExit as e:                 # fix pass: the 863 REFUSED precedent, not RuntimeError
        raised = str(e)
    negative_checks = {
        "raised_SystemExit": raised is not None,
        "message_says_874_REFUSED": raised is not None and "[874] REFUSED" in raised,
        "message_names_the_run_own_gate": raised is not None and "TASK_863" in raised,
        "message_says_not_gradable": raised is not None and "NOT gradable" in raised,
        "no_file_left_behind": (not pre_existed) and (not os.path.exists(target_path)),
    }
    for k, v in negative_checks.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    if raised:
        print(f"    refusal: {raised[:300]}")

    checks = {**{f"pos_{k}": v for k, v in positive_checks.items()},
             **{f"neg_{k}": v for k, v in negative_checks.items()}}
    return {"leg": "iii_pinned_emission_matrix", "passed": all(checks.values()), "checks": checks}


# =============================================================================================
# (B1) a run that REFUSES ITSELF leaves NO provenance -> the checkpoint is UNGRADABLE
# =============================================================================================

def leg_b1_refused_run_emits_nothing(R):
    print("\n=== (B1) degenerate PINNED run refuses itself -> no provenance -> UNGRADABLE ===")
    pin = R("pin_b1.json")
    with open(pin, "w") as fh:
        json.dump({"frame_symbols": PIN_FRAME_SYMBOLS, "n_assets": 8,
                   "train_window": list(GRADE_WINDOW), "db": FIXTURE_DB_REL,
                   "source": "TASK_874 B1 repro pin (863 fixture frame)"}, fh, indent=1)
    ckpt = R("ckpt_degenerate")
    out_json = R("results_degenerate.json")
    # The 863 smoke's own student-side degenerate recipe: a deadband so wide every teacher target
    # is HOLD, imprinted for 80 iters -> total-HOLD policy -> null Sharpe -> FIX-0 refusal.
    rc, text = sh([sys.executable, IMPRINT_PY, "--db", FIXTURE_DB, *PIN_PANEL,
                   "--vector-weight-rebalance", "--imprint-mapping", "weight-native",
                   "--batch-size", "32", "--panel-frame-from-checkpoint", pin,
                   "--require-active-slots", "0", "--vector-rebalance-deadband", "10.0",
                   "--iters", "80", "--seed", "0", "--ckpt-out", ckpt, "--out", out_json],
                  R("log_degenerate.txt"), expect_ok=False)
    prov_path = os.path.join(RUN_DIR, "frame_provenance_degenerate.json")
    res = json.load(open(out_json)) if os.path.exists(out_json) else {}
    checks = {
        "imprint_exit_non_zero": rc != 0,
        "run_refused_itself_863": "[863] REFUSED" in text,
        "artifact_marks_degenerate": ((res.get("sanity") or {}).get("degenerate_replay") is True),
        "ckpt_stamped": os.path.exists(os.path.join(ckpt, "DEGENERATE_REPLAY.json")),
        "NO_provenance_file_written": not os.path.exists(prov_path),
        "emission_was_announced_as_deferred": "deferred to the end of the run" in text,
    }
    for k, v in checks.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")

    # ... and the refused checkpoint cannot be graded: scorecard has nothing to grade it through.
    floor = R("floor_b1.json")
    with open(floor, "w") as fh:
        json.dump({"tag": "toy-b1", "db": FIXTURE_DB_REL, "n_assets": 8,
                   "window": list(GRADE_WINDOW), "floor": {"0": 0.0, "21": 0.0},
                   "ceiling": {"0": 10.0, "21": 10.0}}, fh, indent=2)
    manifest = {
        "schema": "imprint_scorecard_manifest/v1", "name": "874smoke_b1_degenerate",
        "checkpoint": ckpt, "frame_provenance": prov_path,
        "panel_frame_source": "minute", "db": FIXTURE_DB_REL, "n_assets": 8,
        "bar_minutes": GRADE_BAR_MINUTES, "require_active_slots": 0, "eval_seed": 0, "model": {},
        "notes": "TASK_874 fix pass (B1) — grading a REFUSED checkpoint must be impossible.",
        "windows": [{"name": "regime_a", "kind": "regime", "start": GRADE_WINDOW[0],
                     "end": GRADE_WINDOW[1], "floor_artifact": floor,
                     "max_decisions": GRADE_MAX_DECISIONS}],
    }
    manifest_path = R("manifest_b1.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    out_dir = R("battery_out_b1")
    rc2, text2 = sh([sys.executable, SCORECARD_PY, "--manifest", manifest_path,
                     "--out-dir", out_dir], R("log_b1_scorecard.txt"), expect_ok=False)
    sc_path = os.path.join(out_dir, "scorecard_874smoke_b1_degenerate.json")
    sc = json.load(open(sc_path)) if os.path.exists(sc_path) else {}
    gchecks = {
        "scorecard_exit_non_zero": rc2 != 0,
        "scorecard_status_FAILED": sc.get("status") == "FAILED",
        "scorecard_zero_ok": sc.get("n_ok") == 0,
        "refusal_names_the_missing_provenance": any(
            "frame provenance not found" in e for e in sc.get("global_errors", [])),
    }
    for k, v in gchecks.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    checks.update(gchecks)
    return {"leg": "b1_refused_run_emits_nothing", "passed": all(checks.values()),
            "checks": checks}


# =============================================================================================
# (B2) the pad-by-modulo DUPLICATE collapse is REFUSED, not emitted (the panel's repro verbatim)
# =============================================================================================

def leg_b2_duplicate_frame_refused(R):
    print("\n=== (B2) --n-assets 20 on an 8-symbol universe -> duplicate frame, REFUSED ===")
    ckpt = R("ckpt_dup20")
    out_json = R("results_dup20.json")
    rc, text = sh([sys.executable, IMPRINT_PY, "--db", FIXTURE_DB,
                   "--start", "2025-09-14", "--end", "2025-12-12", "--bar-minutes", "240",
                   "--n-assets", "20", "--vector-weight-rebalance", "--iters", "20",
                   "--seed", "0", "--ckpt-out", ckpt, "--out", out_json],
                  R("log_dup20.txt"), expect_ok=False)
    prov_path = os.path.join(RUN_DIR, "frame_provenance_dup20.json")
    checks = {
        "exit_non_zero": rc != 0,
        "refusal_is_874": "[874] REFUSED" in text,
        "names_a_duplicated_symbol": "occupies slots" in text,
        "names_the_modulo_padding_cause": "env_factory.py:434-444" in text,
        "names_the_scorecard_gate_it_would_fail": "scorecard.py:788-792" in text,
        "states_it_refuses_rather_than_normalizes": "does NOT de-duplicate" in text,
        "NO_provenance_file_written": not os.path.exists(prov_path),
        "refused_before_training": "SMOKE PASS" not in text and not os.path.exists(out_json),
    }
    for k, v in checks.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    return {"leg": "b2_duplicate_frame_refused", "passed": all(checks.values()), "checks": checks}


# =============================================================================================
# (B3) an existing provenance file: identical = quiet no-op, differing = REFUSAL, flag = replace
# =============================================================================================

def leg_b3_no_silent_overwrite(R, imp):
    print("\n=== (B3) existing provenance: identical no-op / differing REFUSED / opt-in replace ===")
    from types import SimpleNamespace
    ckpt = R("ckpt_b3")
    os.makedirs(ckpt, exist_ok=True)
    prov_path = os.path.join(RUN_DIR, "frame_provenance_b3.json")
    base = dict(n_assets=8, start="2025-09-14", end="2025-12-12", db=FIXTURE_DB_REL,
                panel_frame_from_checkpoint=None, panel_frame_source="daily",
                overwrite_frame_provenance=False)
    aliases = [{"slot": i, "symbol": s} for i, s in enumerate(FIXTURE_FRAME_SYMBOLS)]
    args = SimpleNamespace(**base)

    written = imp.emit_frame_provenance(ckpt, args, None, aliases)
    first_bytes = open(written, "rb").read()
    first_sha = hashlib.sha256(first_bytes).hexdigest()

    # ---- same run again: IDENTICAL document -> quiet no-op, file untouched ----
    again = imp.emit_frame_provenance(ckpt, args, None, aliases)
    idempotent = open(again, "rb").read() == first_bytes

    # ---- a DIFFERENT document at the same path -> REFUSE, leave the file alone ----
    other = SimpleNamespace(**{**base, "end": "2025-11-30"})
    raised = None
    try:
        imp.emit_frame_provenance(ckpt, other, None, aliases)
    except SystemExit as e:
        raised = str(e)
    untouched = open(prov_path, "rb").read() == first_bytes

    # ---- the explicit opt-in DOES replace, loudly ----
    forced = SimpleNamespace(**{**base, "end": "2025-11-30",
                                "overwrite_frame_provenance": True})
    imp.emit_frame_provenance(ckpt, forced, None, aliases)
    replaced = json.load(open(prov_path))

    # ---- (I2) the replacement is stamped INTO the artifact, and the stamp is not a "difference"
    # ---- (a later identical re-run of the SAME recipe must stay a quiet no-op, not a refusal).
    stamp_free_doc = {k: v for k, v in replaced.items()
                      if k not in ("provenance_replaced", "replaced_provenance_sha256")}
    rerun_raised = None
    try:
        again2 = imp.emit_frame_provenance(ckpt, SimpleNamespace(**{**base, "end": "2025-11-30"}),
                                           None, aliases)
    except SystemExit as e:
        rerun_raised, again2 = str(e), None
    stamp_survived_rerun = json.load(open(prov_path)).get("provenance_replaced") is True

    checks = {
        "first_write_created_the_file": os.path.exists(prov_path),
        "identical_rewrite_is_a_noop": idempotent,
        "differing_write_raised_SystemExit": raised is not None,
        "refusal_is_874": raised is not None and "[874] REFUSED" in raised,
        "refusal_names_the_existing_path": raised is not None and "frame_provenance_b3.json" in raised,
        "refusal_names_the_differing_field": raised is not None and "train_window" in raised,
        "refusal_names_the_opt_in_flag": raised is not None
                                         and "--overwrite-frame-provenance" in raised,
        "refused_write_left_the_file_untouched": untouched,
        "opt_in_flag_replaced_it": replaced["train_window"] == ["2025-09-14", "2025-11-30"],
        # (I2) the artifact, not just the log, records that a DIFFERENT provenance stood here
        "i2_document_records_the_replacement": replaced.get("provenance_replaced") is True,
        "i2_document_digests_what_it_replaced": replaced.get("replaced_provenance_sha256")
                                                == first_sha,
        "i2_stamp_is_the_ONLY_added_field": set(stamp_free_doc) == {
            "frame_symbols", "n_assets", "source", "train_window", "db", "checkpoint",
            "panel_frame_source"},
        # the stamp describes the FILE's history, not the frame -> an identical re-run stays a
        # quiet no-op (it must not degrade into a refusal on every subsequent run)
        "i2_identical_rerun_over_a_stamped_file_is_a_noop": rerun_raised is None
                                                            and again2 == prov_path,
        "i2_stamp_survives_that_rerun": stamp_survived_rerun,
    }
    for k, v in checks.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    if raised:
        print(f"    refusal: {raised[:300]}")
    return {"leg": "b3_no_silent_overwrite", "passed": all(checks.values()), "checks": checks}


# =============================================================================================
# (I1) a provenance REFUSAL can never move the git-tracked parity baseline
# =============================================================================================

def leg_i1a_promoting_run_refuses_without_moving_the_baseline(R):
    print("\n=== (I1a) canon --write-parity-baseline + conflicting provenance -> exit 1, "
          "baseline byte-identical ===")
    planted = conflicting_provenance_doc("TASK_874 I1a planted conflict")
    with open(CANON_PROV, "w") as fh:
        json.dump(planted, fh, indent=1)
    planted_sha = sha256_file(CANON_PROV)
    sha_before = sha256_file(PARITY_BASELINE)
    print(f"    results_fixture_smoke.json sha256 BEFORE: {sha_before}")
    rc, text = sh([sys.executable, IMPRINT_PY, *CANON_PROMOTE_ARGV],
                  R("log_i1a_promote_refused.txt"), expect_ok=False)
    sha_after = sha256_file(PARITY_BASELINE)
    print(f"    results_fixture_smoke.json sha256 AFTER : {sha_after}")
    checks = {
        "exit_non_zero": rc != 0,
        "refusal_is_874": "[874] REFUSED" in text,
        "refusal_names_the_existing_file": "frame_provenance_smoke.json" in text,
        "promotion_never_ran": "parity baseline PROMOTED" not in text,
        "baseline_sha256_unchanged": sha_before is not None and sha_before == sha_after,
        "planted_provenance_untouched": sha256_file(CANON_PROV) == planted_sha,
    }
    for k, v in checks.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    return {"leg": "i1a_promoting_run_refuses_without_moving_the_baseline",
            "passed": all(checks.values()), "checks": checks,
            "baseline_sha256_before": sha_before, "baseline_sha256_after": sha_after}


def leg_i1b_refusal_precedes_promotion(R, imp):
    """The RACE the fix actually closes: the conflicting file appears AFTER run()'s own
    pre-training dry run has already looked. Drives imp.main() with a stub run() and a
    NON-MUTATING promote spy, so the assertion is pure ORDERING and this leg cannot move the
    baseline even if it fails."""
    print("\n=== (I1b) provenance validated BEFORE promote_parity_baseline is reached ===")
    sha_before = sha256_file(PARITY_BASELINE)
    calls = []
    doc = {"frame_symbols": FIXTURE_FRAME_SYMBOLS, "n_assets": 8,
           "source": "TASK_874 I1b stub emission", "train_window": ["2025-09-14", "2025-12-12"],
           "db": FIXTURE_DB_REL, "checkpoint": "experiments/832_linear_imprint/ckpt_smoke",
           "panel_frame_source": "minute"}

    def stub_run(args, out_path, tag, provenance_sink=None):
        calls.append("run")
        with open(CANON_PROV, "w") as fh:                    # appears mid-run, post-dry-run
            json.dump(conflicting_provenance_doc("TASK_874 I1b mid-run conflict"), fh, indent=1)
        provenance_sink(CANON_PROV, doc)
        return {"stub_result": True}

    def spy_promote(*a, **kw):                               # records, NEVER mutates
        calls.append("promote")

    real_run, real_promote = imp.run, imp.promote_parity_baseline
    old_argv = sys.argv
    raised = None
    try:
        imp.run, imp.promote_parity_baseline = stub_run, spy_promote
        sys.argv = ["imprint.py", *CANON_PROMOTE_ARGV]
        imp.main()
    except SystemExit as e:
        raised = str(e)
    finally:
        imp.run, imp.promote_parity_baseline = real_run, real_promote
        sys.argv = old_argv
    checks = {
        "run_was_reached": calls[:1] == ["run"],
        "raised_SystemExit": raised is not None,
        "refusal_is_874": raised is not None and "[874] REFUSED" in raised,
        "promote_parity_baseline_NEVER_called": "promote" not in calls,
        "baseline_sha256_unchanged": sha256_file(PARITY_BASELINE) == sha_before,
    }
    for k, v in checks.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    print(f"    call order: {calls}")
    return {"leg": "i1b_refusal_precedes_promotion", "passed": all(checks.values()),
            "checks": checks, "call_order": list(calls)}


def legs_i1_baseline_never_moves(R, imp):
    """Both I1 sub-legs, with the shared canon provenance path saved and restored around them."""
    had = os.path.exists(CANON_PROV)
    saved = open(CANON_PROV, "rb").read() if had else None
    try:
        return [leg_i1a_promoting_run_refuses_without_moving_the_baseline(R),
                leg_i1b_refusal_precedes_promotion(R, imp)]
    finally:
        if saved is not None:
            with open(CANON_PROV, "wb") as fh:
                fh.write(saved)
        elif os.path.exists(CANON_PROV):
            os.unlink(CANON_PROV)


def main():
    ap = argparse.ArgumentParser(description="TASK_874 frame-provenance-emitter fixture smoke")
    ap.add_argument("--keep", action="store_true", help="keep a previous run dir's artifacts")
    args = ap.parse_args()
    # TASK_873: verify the live fixture against fixtures/FIXTURE_IDENTITY.json BEFORE any work.
    fixture_identity.verify_or_die(FIXTURE_DB, label="[874 fixture-identity]")
    if not args.keep and os.path.isdir(RUN_DIR):
        shutil.rmtree(RUN_DIR)
    os.makedirs(RUN_DIR, exist_ok=True)
    R = lambda *p: os.path.join(RUN_DIR, *p)                              # noqa: E731
    report = {"legs": []}
    t0 = time.time()

    leg1 = leg_i_derived_emit_and_battery(R)
    report["legs"].append(leg1)
    if leg1["passed"]:
        report["legs"].append(leg_ii_cross_checkpoint_refused(R, leg1["prov_path"]))
    else:
        print("[874] SKIPPING (ii): leg (i) did not pass, no provenance file to reuse")

    imp = _imprint_module()
    report["legs"].append(leg_iii_pinned_emission_matrix(R, imp))
    report["legs"].append(leg_b3_no_silent_overwrite(R, imp))
    report["legs"].append(leg_b2_duplicate_frame_refused(R))
    report["legs"].append(leg_b1_refused_run_emits_nothing(R))
    report["legs"].extend(legs_i1_baseline_never_moves(R, imp))

    report["all_passed"] = all(l["passed"] for l in report["legs"])
    report["elapsed_s"] = round(time.time() - t0, 1)
    with open(R("report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print("\n" + "=" * 84)
    for l in report["legs"]:
        print(f"  {'PASS' if l['passed'] else 'FAIL'}  {l['leg']}")
    print(f"TASK_874 frame-provenance smoke: "
          f"{'ALL LEGS PASS' if report['all_passed'] else 'FAILURES'} in {report['elapsed_s']}s "
          f"-> {R('report.json')}")
    print("=" * 84)
    raise SystemExit(0 if report["all_passed"] else 1)


if __name__ == "__main__":
    main()
