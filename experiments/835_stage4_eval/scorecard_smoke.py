#!/usr/bin/env python3
"""TASK_862 fixture smoke for ``scorecard.py`` — the factory-scale acceptance run.

Builds a self-contained toy battery on ``fixtures/trading_fixture.db`` (NEVER the read-only
``trading_data.db``) in a work directory outside the repo, then exercises the three legs the
task's acceptance names:

  (ii)  a 2-window toy manifest runs END-TO-END and emits a SCHEMA-VALID scorecard
        (``scorecard.validate_scorecard``), and a re-run is BYTE-IDENTICAL (json + md).
  (iii) fail-loud: a MISSING FLOOR ARTIFACT and a MISMATCHED FRAME each produce a FAILED
        row in the emitted artifact AND a non-zero exit code (no partial-success masquerade).

Plus the post-review hardening legs (2026-07-27 adversarial panel):

  (iv)  a MALFORMED floor (non-numeric value, explicit null) is a FAILED row — not a crash,
        not a silently-zero floor — and a LATE window failure never discards the windows that
        already completed: the artifact still carries window 1's numbers.
  (v)   the operator is OBSERVED, not asserted: mutating eval_vector's own emitted result
        (decision_rule top-m, a different checkpoint, a different position_size_pct, a moved
        `sims`/`leaf_eval`/`model_size` default) is REJECTED by the same check the battery
        uses, and rejection aborts the rest of the battery instead of grading more windows
        under the wrong operator.
  (vi)  manifest-level refusals (unknown key, bad model_size, a '..' window name, a duplicate
        name, start >= end, an UNDECLARED/unrecognised panel_frame_source, truncated JSON) exit 2
        with a clean message, never a traceback.

And the D30 rider (c) legs — the panel's PRICE SOURCE is plumbed, not assumed (2026-07-30):

  (vii) a MINUTE-declared manifest reaches eval_vector with ``--panel-frame-source minute``, the
        emitted result's ``panel_frame.panel_frame_source`` AGREES, and the scorecard records it in
        the header and in every row. Negative control in the same leg: a DAILY-declared manifest
        emits NO ``--panel-frame-source`` flag at all — its command line is the 848 read of record
        byte-for-byte (flag-gated, default OFF), and the minute argv differs from it by exactly
        those two tokens.
  (viii) the guards, each shown FIRING and each with the negative control where it must NOT fire:
        a declared/actual source mismatch in EITHER direction is refused; a minute-declared
        manifest whose result carries NO source key is refused (absence means DAILY — the
        convention is read correctly, not mistaken for a missing field); a daily-declared manifest
        whose result carries no key is ACCEPTED (that IS the shipped 853 artifact); an unrecognised
        emitted source is refused; and the source/stride coherence rule refuses 'daily' off 1440
        and 'minute' under a stride < 1 while accepting the coherent pairs.

Everything the toy battery needs is generated here — an untrained checkpoint (via the shipped
``eval_vector.py --smoke`` save→load roundtrip, written to the work dir so the committed
``results_fixture_smoke.json`` parity baseline is never touched), a frame provenance over the
fixture's 8-symbol minute universe (all have full ``ohlcv_daily`` coverage), toy per-window
floor artifacts in the ``runs/831_ext`` convention, and the manifest itself.

Toy window names are deliberately NOT the four DEC-2 names, so eval_vector's own 831-EXT
per-window block can never be confused with the pinned toy floors.

Usage::

    python experiments/835_stage4_eval/scorecard_smoke.py [--work-dir DIR] [--keep]

Untrained numbers are near-noise by construction — this is a plumbing/schema proof, no
threshold is applied. Exit 0 = PASS.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

sys.path.insert(0, _HERE)
import scorecard as sc_mod                                        # noqa: E402

EVAL_VECTOR = os.path.join(_HERE, "eval_vector.py")
SCORECARD = os.path.join(_HERE, "scorecard.py")
FIXTURE_DB_REL = os.path.join("fixtures", "trading_fixture.db")

# eval_vector.SMOKE_SYMBOLS — the fixture's minute-table universe; all 8 also carry full
# ohlcv_daily history in the fixture, so the fixed-frame daily panel has 8 ACTIVE slots.
FRAME_SYMBOLS = ["XBT/USD", "ETH/USD", "LTC/USD", "XMR/USD",
                 "ZEC/USD", "ETC/USD", "DASH/USD", "REP/USD"]
N_ASSETS = len(FRAME_SYMBOLS)
TOY_WINDOWS = [("toy_a", "2024-01-01", "2024-06-30"),
               ("toy_b", "2024-07-01", "2024-12-31")]
MAX_DECISIONS = 4                      # fixture scale: plumbing proof, not a measurement

# The fixture's MINUTE table spans 2025-09-13..2025-12-12 with 27,181 distinct minute timestamps
# (the daily table reaches back to 2013, which is why the daily toy windows are in 2024 and the
# minute ones cannot be). At a 60-minute stride that is ~453 panel rows > warm 60 + decisions —
# eval_vector's own SMOKE_BAR_MINUTES choice, for the same reason. 1440 would yield 19 rows.
MINUTE_WINDOWS = [("min_a", "2025-09-13", "2025-12-12")]
MINUTE_BAR_MINUTES = 60


def _run(argv, **kw):
    print(f"    $ {' '.join(str(a) for a in argv[1:])}")
    return subprocess.run([str(a) for a in argv], cwd=_REPO, capture_output=True,
                          text=True, **kw)


def ensure_checkpoint(work):
    """An untrained n_assets=8 checkpoint via the SHIPPED eval_vector smoke roundtrip.
    Written under the work dir (never the module's committed smoke artifacts)."""
    ckpt = os.path.join(work, "ckpt")
    if os.path.exists(os.path.join(ckpt, "trainer_metadata.json")):
        print("[smoke] reusing checkpoint from a previous smoke run")
        return ckpt
    out = os.path.join(work, "ckpt_build")
    os.makedirs(out, exist_ok=True)
    print("[smoke] building an untrained fixture checkpoint via eval_vector --smoke ...")
    p = _run([sys.executable, EVAL_VECTOR, "--smoke", "--max-decisions", 2,
              "--smoke-ckpt-dir", ckpt, "--out-dir", out])
    if p.returncode != 0:
        print(p.stdout[-4000:])
        print(p.stderr[-4000:])
        raise SystemExit("[smoke] FAILED: could not build the fixture checkpoint")
    return ckpt


def write_inputs(work, ckpt, *, frame=None, floors_present=True, windows=None, name="toysmoke",
                 floor_overrides=None, panel_frame_source="daily", bar_minutes=None,
                 require_active_slots=None):
    """Write provenance + toy floor artifacts + manifest. Returns the manifest path.

    ``floor_overrides`` maps a window name to the literal ``floor`` object to write (used by
    leg (iv) to plant a non-numeric / explicitly-null floor).

    ``panel_frame_source`` is written EXPLICITLY (D30 rider (c) gives it no default; a manifest
    that omits it is refused — leg (vi) proves that), and ``bar_minutes`` is omitted from the
    manifest when None so the default (1440, the 853 daily rule) is exercised too."""
    frame = FRAME_SYMBOLS if frame is None else frame
    windows = TOY_WINDOWS if windows is None else windows
    floor_overrides = floor_overrides or {}
    prov_path = os.path.join(work, f"frame_provenance_{name}.json")
    with open(prov_path, "w") as fh:
        json.dump({"frame_symbols": frame,
                   "n_assets": len(frame),
                   "checkpoint": ckpt,
                   "db": FIXTURE_DB_REL,
                   "source": "TASK_862 fixture smoke — fixture minute universe, slot order fixed"},
                  fh, indent=2)
    win_specs = []
    for wname, start, end in windows:
        floor_path = os.path.join(work, f"floor_{name}_{wname}.json")
        if floors_present:
            with open(floor_path, "w") as fh:
                json.dump({"tag": f"toy-{wname}", "db": FIXTURE_DB_REL, "n_assets": len(frame),
                           "n_decisions": MAX_DECISIONS, "window": [start, end],
                           "floor": floor_overrides.get(wname, {"0": 0.0, "21": 0.0}),
                           "ceiling": {"0": 10.0, "21": 10.0}}, fh, indent=2)
        else:
            floor_path = os.path.join(work, f"floor_{name}_{wname}_ABSENT.json")
        win_specs.append({"name": wname, "kind": "regime", "start": start, "end": end,
                          "floor_artifact": floor_path, "max_decisions": MAX_DECISIONS})
    manifest_path = os.path.join(work, f"manifest_{name}.json")
    obj = {"schema": sc_mod.MANIFEST_SCHEMA,
           "name": name,
           "checkpoint": ckpt,
           "frame_provenance": prov_path,
           "panel_frame_source": panel_frame_source,
           "db": FIXTURE_DB_REL,
           "n_assets": N_ASSETS,
           "eval_seed": 0,
           "notes": "TASK_862 fixture smoke (toy battery — no thresholds applied)",
           "windows": win_specs}
    if bar_minutes is not None:
        obj["bar_minutes"] = bar_minutes
    # D30 rider (b) is an optional manifest knob with a PINNED default; omitted here so the default
    # path (always emitted into argv, echoed in the header, observed back out of every row) is what
    # the legs exercise.
    if require_active_slots is not None:
        obj["require_active_slots"] = require_active_slots
    with open(manifest_path, "w") as fh:
        json.dump(obj, fh, indent=2)
    return manifest_path


def run_scorecard(manifest, out_dir):
    p = _run([sys.executable, SCORECARD, "--manifest", manifest, "--out-dir", out_dir])
    print(p.stdout[-3000:])
    if p.returncode != 0:
        print(p.stderr[-2000:])
    return p


def _read_bytes(path):
    with open(path, "rb") as fh:
        return fh.read()


def leg_ii(work, ckpt):
    print("\n" + "=" * 84)
    print("[smoke] LEG (ii) — 2-window toy manifest end-to-end + schema + byte-identical re-run")
    print("=" * 84)
    manifest = write_inputs(work, ckpt)
    out_dir = os.path.join(work, "out")
    p1 = run_scorecard(manifest, out_dir)
    if p1.returncode != 0:
        print(p1.stderr[-4000:])
        raise SystemExit("[smoke] FAIL (ii): first scorecard run exited non-zero")
    json_path = os.path.join(out_dir, "scorecard_toysmoke.json")
    md_path = os.path.join(out_dir, "scorecard_toysmoke.md")
    b1, m1 = _read_bytes(json_path), _read_bytes(md_path)

    sc = json.loads(b1)
    sc_mod.validate_scorecard(sc)                    # raises on any schema violation
    if sc["status"] != "OK" or sc["n_failed"]:
        raise SystemExit(f"[smoke] FAIL (ii): status={sc['status']} n_failed={sc['n_failed']}")
    if [w["name"] for w in sc["windows"]] != [w[0] for w in TOY_WINDOWS]:
        raise SystemExit("[smoke] FAIL (ii): window rows do not match the manifest")
    for row in sc["windows"]:
        for tier in sc_mod.TIERS:
            if row["tiers"][tier]["teacher_floor"] is None:
                raise SystemExit(f"[smoke] FAIL (ii): {row['name']} tier {tier} carries no "
                                 f"on-window teacher floor")
        if row["regime_floor"] is None:
            raise SystemExit(f"[smoke] FAIL (ii): {row['name']} carries no regime floor block")
        # D30 rider (c): the DAILY direction, end to end — the row must OBSERVE 'daily' out of a
        # result whose panel_frame block does not carry the key at all (the shipped 853 artifact).
        if row["panel"]["panel_frame_source"] != "daily":
            raise SystemExit(f"[smoke] FAIL (ii): {row['name']} panel_frame_source is "
                             f"{row['panel']['panel_frame_source']!r}, expected 'daily'")
    if sc["operator"]["panel_frame_source"] != "daily" or sc["bar_minutes"] != 1440:
        raise SystemExit(f"[smoke] FAIL (ii): header source/stride "
                         f"{sc['operator']['panel_frame_source']!r}/{sc['bar_minutes']} "
                         f"!= 'daily'/1440")
    # FIX PASS — D30 rider (b) threaded the way rider (c) threads the source: PINNED in the header,
    # emitted into argv, OBSERVED back out of every row. Before this it was neither pinned nor
    # observed, so if eval_vector's shared default moved, the read of record could not tell.
    want_floor = sc_mod.REQUIRE_ACTIVE_SLOTS
    if sc["require_active_slots"] != want_floor:
        raise SystemExit(f"[smoke] FAIL (ii): header require_active_slots "
                         f"{sc['require_active_slots']!r} != {want_floor!r}")
    if sc["operator"]["panel_frame_pin"]["require_active_slots"] != want_floor:
        raise SystemExit("[smoke] FAIL (ii): the breadth floor is not in panel_frame_pin")
    if f"--require-active-slots {want_floor!r}" not in sc["operator"]["argv_template"]:
        raise SystemExit(f"[smoke] FAIL (ii): argv_template does not carry the breadth floor: "
                         f"{sc['operator']['argv_template']}")
    for row in sc["windows"]:
        if row["panel"]["require_active_slots"] != want_floor:
            raise SystemExit(f"[smoke] FAIL (ii): {row['name']} OBSERVED breadth floor "
                             f"{row['panel']['require_active_slots']!r} != {want_floor!r} — the "
                             f"floor actually applied is not the one pinned")
        # ... and the cell-honesty numbers are observed back out too, so a row can never read as a
        # fully-live grid when it is not (the daily pin emits them as of the fix pass).
        if row["panel"]["not_real_decision_cell_pct"] is None:
            raise SystemExit(f"[smoke] FAIL (ii): {row['name']} reports no "
                             f"not_real_decision_cell_pct")
    if "breadth floor" not in _read_bytes(md_path).decode():
        raise SystemExit("[smoke] FAIL (ii): the MD does not state the breadth floor")
    print(f"[smoke] (ii) breadth floor {want_floor} pinned in argv + header, OBSERVED per row; "
          f"not-real cell share " + ", ".join(
              f"{r['name']}={r['panel']['not_real_decision_cell_pct']}%" for r in sc["windows"]))
    print(f"[smoke] (ii) schema-valid; {sc['n_ok']}/{sc['n_windows']} windows OK; "
          f"net@21 " + ", ".join(f"{r['name']}={r['tiers']['21']['sharpe']}"
                                 for r in sc["windows"]))

    p2 = run_scorecard(manifest, out_dir)
    if p2.returncode != 0:
        raise SystemExit("[smoke] FAIL (ii): re-run exited non-zero")
    b2, m2 = _read_bytes(json_path), _read_bytes(md_path)
    if b1 != b2:
        raise SystemExit("[smoke] FAIL (ii): scorecard JSON is NOT byte-identical on re-run")
    if m1 != m2:
        raise SystemExit("[smoke] FAIL (ii): scorecard MD is NOT byte-identical on re-run")
    print(f"[smoke] (ii) PASS — re-run byte-identical (json {len(b1)} B, md {len(m1)} B)")


def leg_iii_missing_floor(work, ckpt):
    print("\n" + "=" * 84)
    print("[smoke] LEG (iii)a — missing floor artifact ⇒ FAILED row + non-zero exit")
    print("=" * 84)
    manifest = write_inputs(work, ckpt, floors_present=False, windows=TOY_WINDOWS[:1],
                            name="nofloor")
    out_dir = os.path.join(work, "out_nofloor")
    p = run_scorecard(manifest, out_dir)
    if p.returncode == 0:
        raise SystemExit("[smoke] FAIL (iii)a: exit code was 0 on a missing floor artifact")
    sc = json.loads(_read_bytes(os.path.join(out_dir, "scorecard_nofloor.json")))
    sc_mod.validate_scorecard(sc)
    row = sc["windows"][0]
    if sc["status"] != "FAILED" or row["status"] != "FAILED":
        raise SystemExit("[smoke] FAIL (iii)a: no FAILED row emitted")
    if "floor artifact not found" not in row["reason"]:
        raise SystemExit(f"[smoke] FAIL (iii)a: unexpected reason {row['reason']!r}")
    if any(k in row for k in ("tiers", "n_decisions")):
        raise SystemExit("[smoke] FAIL (iii)a: a FAILED row carries scored keys")
    print(f"[smoke] (iii)a PASS — exit {p.returncode}, row reason: {row['reason']}")


def leg_iii_bad_frame(work, ckpt):
    print("\n" + "=" * 84)
    print("[smoke] LEG (iii)b — mismatched frame ⇒ FAILED rows + non-zero exit (refusal)")
    print("=" * 84)
    manifest = write_inputs(work, ckpt, frame=FRAME_SYMBOLS[:-1], name="badframe")
    out_dir = os.path.join(work, "out_badframe")
    p = run_scorecard(manifest, out_dir)
    if p.returncode == 0:
        raise SystemExit("[smoke] FAIL (iii)b: exit code was 0 on a mismatched frame")
    sc = json.loads(_read_bytes(os.path.join(out_dir, "scorecard_badframe.json")))
    sc_mod.validate_scorecard(sc)
    if sc["status"] != "FAILED" or sc["n_ok"] != 0 or not sc["global_errors"]:
        raise SystemExit("[smoke] FAIL (iii)b: expected a global refusal with all rows FAILED")
    if not all(r["status"] == "FAILED" for r in sc["windows"]):
        raise SystemExit("[smoke] FAIL (iii)b: not every window row is FAILED")
    if not any("frame/provenance mismatch" in e for e in sc["global_errors"]):
        raise SystemExit(f"[smoke] FAIL (iii)b: unexpected errors {sc['global_errors']}")
    if os.path.exists(os.path.join(out_dir, "windows")):
        raise SystemExit("[smoke] FAIL (iii)b: a window was executed despite the refusal")
    print(f"[smoke] (iii)b PASS — exit {p.returncode}, refused: {sc['global_errors'][0]}")


def leg_iv_malformed_floor(work, ckpt):
    """A malformed floor value is a FAILED row, and a LATE failure never discards the windows
    that already completed (the panel's top finding: only WindowFailure used to be caught, so
    a `float('n/a')` ValueError destroyed the whole artifact)."""
    print("\n" + "=" * 84)
    print("[smoke] LEG (iv) — malformed/null floor ⇒ FAILED rows, completed windows PRESERVED")
    print("=" * 84)
    windows = [TOY_WINDOWS[0], TOY_WINDOWS[1], ("toy_c", "2023-01-01", "2023-06-30")]
    manifest = write_inputs(work, ckpt, windows=windows, name="badfloor",
                            floor_overrides={"toy_b": {"0": "n/a", "21": "n/a"},
                                             "toy_c": {"0": None, "21": None}})
    out_dir = os.path.join(work, "out_badfloor")
    p = run_scorecard(manifest, out_dir)
    if p.returncode == 0:
        raise SystemExit("[smoke] FAIL (iv): exit code was 0 on malformed floors")
    json_path = os.path.join(out_dir, "scorecard_badfloor.json")
    if not os.path.exists(json_path):
        raise SystemExit("[smoke] FAIL (iv): NO artifact was written — a late failure destroyed "
                         "the whole battery")
    sc = json.loads(_read_bytes(json_path))
    sc_mod.validate_scorecard(sc)
    rows = {r["name"]: r for r in sc["windows"]}
    if rows["toy_a"]["status"] != "OK" or rows["toy_a"]["tiers"]["21"]["sharpe"] is None:
        raise SystemExit("[smoke] FAIL (iv): the completed window's numbers were lost")
    for bad, needle in (("toy_b", "is not a number"), ("toy_c", "is None")):
        if rows[bad]["status"] != "FAILED":
            raise SystemExit(f"[smoke] FAIL (iv): {bad} is not FAILED")
        if needle not in rows[bad]["reason"]:
            raise SystemExit(f"[smoke] FAIL (iv): {bad} reason {rows[bad]['reason']!r} does not "
                             f"name the malformation ({needle})")
    if not os.path.exists(os.path.join(out_dir, "scorecard_badfloor.md")):
        raise SystemExit("[smoke] FAIL (iv): no MD render was written")
    # the panel's FIX 8: the honesty flags must be visible in the MD, not only the JSON
    md = _read_bytes(os.path.join(out_dir, "scorecard_badfloor.md")).decode()
    for needle in ("CAPPED@4", "CROSS-CFG", "Configuration delta"):
        if needle not in md:
            raise SystemExit(f"[smoke] FAIL (iv): MD does not surface {needle!r}")
    print(f"[smoke] (iv) PASS — exit {p.returncode}; toy_a preserved "
          f"(net@21={rows['toy_a']['tiers']['21']['sharpe']}), toy_b/toy_c FAILED with reasons")


def leg_v_operator_observed(work, ckpt):
    """The operator must be OBSERVED out of eval_vector's own result, not asserted from our
    argv. Mutate a REAL emitted result and check the harness rejects every drift."""
    print("\n" + "=" * 84)
    print("[smoke] LEG (v) — operator observed + asserted; drift aborts the battery")
    print("=" * 84)
    manifest_path = os.path.join(work, "manifest_toysmoke.json")
    manifest = sc_mod.load_manifest(manifest_path)
    win_dir = os.path.join(work, "out", "windows", "toy_a")
    found = [f for f in sorted(os.listdir(win_dir)) if f.startswith("results_")
             and f.endswith(".json")]
    if not found:
        raise SystemExit("[smoke] FAIL (v): leg (ii) left no eval_vector result to inspect")
    with open(os.path.join(win_dir, found[0])) as fh:
        base = json.load(fh)
    unpinned = sc_mod._check_operator(base, manifest)
    if unpinned:
        raise SystemExit(f"[smoke] FAIL (v): the real 848 read reports unpinned operator "
                         f"fields {unpinned} — the pin is out of date")
    print("[smoke] (v) baseline: the real read satisfies the full 848 operator pin")

    def mutate(**path_vals):
        res = json.loads(json.dumps(base))
        for dotted, val in path_vals.items():
            head, _, key = dotted.rpartition(".")
            tgt = res if not head else res[head]
            if val is sc_mod:                       # sentinel: delete the key
                tgt.pop(key, None)
            else:
                tgt[key] = val
        return res

    cases = {
        "decision-rule top-m": {"operator.decision_rule": "top-m", "operator.decision_m": 2},
        "leaf_eval true-env": {"operator.leaf_eval": "true-env"},
        "prior_weight 0.0": {"operator.prior_weight": 0.0},
        "sims default moved": {"operator.sims": 8},
        "discount moved": {"operator.discount": 0.5},
        "model_size large": {"operator.model_size": "large"},
        "rebalance off": {"operator.vector_weight_rebalance": False},
        "other checkpoint": {"checkpoint": "/some/other/checkpoint"},
        "position_size_pct 0.9": {"panel.position_size_pct": 0.9},
        "warm 31": {"panel.warm": 31},
        "operator block gone": {"operator": sc_mod},
    }
    for label, muts in cases.items():
        try:
            sc_mod._check_operator(mutate(**muts), manifest)
        except sc_mod.OperatorPinFailure as e:
            print(f"[smoke] (v) rejected {label:<22} — {str(e)[:110]}")
            continue
        raise SystemExit(f"[smoke] FAIL (v): {label} was ACCEPTED as the 848 operator")

    extra = mutate(**{"operator.search_teacher_head": True})
    got = sc_mod._check_operator(extra, manifest)
    if got != ["search_teacher_head"]:
        raise SystemExit(f"[smoke] FAIL (v): a NEW operator field should be recorded as "
                         f"unpinned, got {got!r}")
    print("[smoke] (v) a new (unpinned) operator field is recorded, not silently dropped")

    # …and a pin failure ABORTS the battery instead of burning the remaining windows.
    real_run_window = sc_mod.run_window
    calls = []

    def boom(manifest_, window, *a, **kw):
        calls.append(window["name"])
        raise sc_mod.OperatorPinFailure("synthetic drift for the abort leg")
    sc_mod.run_window = boom
    try:
        out_dir = os.path.join(work, "out_abort")
        sc, jp, _ = sc_mod.run_battery(manifest_path, out_dir, verbose=False)
    finally:
        sc_mod.run_window = real_run_window
    sc_mod.validate_scorecard(sc)
    if calls != ["toy_a"]:
        raise SystemExit(f"[smoke] FAIL (v): the battery kept running after an operator pin "
                         f"failure (windows attempted: {calls})")
    if sc["status"] != "FAILED" or sc["n_ok"] != 0:
        raise SystemExit("[smoke] FAIL (v): abort did not mark every window FAILED")
    if "ABORTED" not in sc["windows"][1]["reason"]:
        raise SystemExit("[smoke] FAIL (v): the un-run window does not say why it did not run")
    if not os.path.exists(jp):
        raise SystemExit("[smoke] FAIL (v): no artifact written on an aborted battery")
    print(f"[smoke] (v) PASS — abort after {calls}, both rows FAILED, artifact still written")

    # a NON-WindowFailure exception is a FAILED row too (never a lost artifact)
    def kaboom(manifest_, window, *a, **kw):
        raise ValueError("synthetic non-WindowFailure explosion")
    sc_mod.run_window = kaboom
    try:
        sc2, jp2, _ = sc_mod.run_battery(manifest_path, os.path.join(work, "out_boom"),
                                         verbose=False)
    finally:
        sc_mod.run_window = real_run_window
    sc_mod.validate_scorecard(sc2)
    if sc2["n_failed"] != 2 or "unexpected ValueError" not in sc2["windows"][0]["reason"]:
        raise SystemExit("[smoke] FAIL (v): a non-WindowFailure exception was not contained")
    if not os.path.exists(jp2):
        raise SystemExit("[smoke] FAIL (v): no artifact written on an unexpected exception")
    print("[smoke] (v) PASS — a raw ValueError becomes a FAILED row, artifact still written")

    # a row that violates the schema is DEGRADED to FAILED, never suppressed — the writer's
    # self-check must not be able to destroy an emitted battery either.
    def schema_violating_row(manifest_, window, *a, **kw):
        return {"name": window["name"], "kind": window["kind"], "status": "OK",
                "window": [window["start"], window["end"]]}      # every scored key missing
    sc_mod.run_window = schema_violating_row
    try:
        sc3, jp3, mp3 = sc_mod.run_battery(manifest_path, os.path.join(work, "out_degrade"),
                                           verbose=False)
    finally:
        sc_mod.run_window = real_run_window
    sc_mod.validate_scorecard(sc3)                # the DEGRADED artifact is still schema-valid
    if sc3["self_check"]["ok"] or not sc3["self_check"]["errors"]:
        raise SystemExit("[smoke] FAIL (v): the self-check did not record the violation")
    if sc3["status"] != "FAILED" or sc3["n_failed"] != 2:
        raise SystemExit("[smoke] FAIL (v): degraded rows did not fail the scorecard")
    if "self-check" not in sc3["windows"][0]["reason"]:
        raise SystemExit("[smoke] FAIL (v): a degraded row does not say why")
    if not (os.path.exists(jp3) and os.path.exists(mp3)):
        raise SystemExit("[smoke] FAIL (v): the self-check suppressed the artifact")
    print(f"[smoke] (v) PASS — schema-violating rows degraded to FAILED "
          f"({sc3['self_check']['degraded_rows']}), artifact still written")


_DROP = object()          # sentinel: leg (vi) DELETES the key instead of patching its value

BAD_MANIFESTS = {
    "unknown_key": ({"eval_seeds": 3}, "unknown key"),
    # D30 rider (c): a frame-pinned manifest that does not DECLARE its price source must refuse,
    # not default to daily by silence.
    "undeclared_source": ({"panel_frame_source": _DROP}, "'panel_frame_source' is REQUIRED"),
    "null_source": ({"panel_frame_source": None}, "'panel_frame_source' is REQUIRED"),
    "unknown_source": ({"panel_frame_source": "hourly"}, "expected one of ['daily', 'minute']"),
    "nonstring_source": ({"panel_frame_source": 1440}, "must be a string"),
    # …and an incoherent source/stride pair (daily reads already-daily rows) refuses too.
    "incoherent_stride": ({"bar_minutes": 60}, "INCOHERENT panel frame source/stride"),
    "bad_stride": ({"bar_minutes": 0}, "'bar_minutes' must be a positive int"),
    # FIX PASS — bar_minutes became a manifest knob while the header still called the operator
    # "FIXED in code". {"panel_frame_source": "minute", "bar_minutes": 1} used to load CLEAN and
    # would have graded the read of record on a ONE-MINUTE bar with status OK (bar_minutes is also
    # the Sharpe periodization, and every floor/corridor artifact is a daily-bar object). Bounded.
    "sub_hourly_minute_stride": ({"panel_frame_source": "minute", "bar_minutes": 1},
                                 "OUT-OF-RANGE stride"),
    "supra_daily_minute_stride": ({"panel_frame_source": "minute", "bar_minutes": 2880},
                                  "OUT-OF-RANGE stride"),
    # FIX PASS — the breadth floor is a bounded knob too: eval_vector used to FAIL OPEN on every
    # one of these, recording a floor as though one were in force.
    "negative_floor": ({"require_active_slots": -1}, "is negative"),
    "bool_floor": ({"require_active_slots": True}, "not a toggle"),
    "string_floor": ({"require_active_slots": "0"}, "must be a number"),
    "infinitesimal_floor": ({"require_active_slots": 1e-300}, "too small to refuse anything"),
    "fractional_slot_count": ({"require_active_slots": 1.5}, "fractional slot count"),
    "bad_model_size": ({"model": {"model_size": "gigantic"}}, "model_size"),
    "dotdot_window": ({"windows": [{"name": "..", "kind": "oos", "start": "2024-01-01",
                                    "end": "2024-06-30"}]}, "name"),
    "dup_window": ({"windows": [{"name": "w", "kind": "oos", "start": "2024-01-01",
                                 "end": "2024-06-30"},
                                {"name": "w", "kind": "oos", "start": "2024-01-01",
                                 "end": "2024-06-30"}]}, "duplicate"),
    "backwards_span": ({"windows": [{"name": "w", "kind": "oos", "start": "2024-06-30",
                                     "end": "2024-01-01"}]}, "not before"),
}


def leg_vi_manifest_refusals(work, ckpt):
    """Manifest-level errors exit 2 with a clean message — never an uncaught traceback."""
    print("\n" + "=" * 84)
    print("[smoke] LEG (vi) — manifest refusals exit 2 cleanly (no traceback)")
    print("=" * 84)
    base_path = os.path.join(work, "manifest_toysmoke.json")
    with open(base_path) as fh:
        base = json.load(fh)
    for label, (patch, needle) in BAD_MANIFESTS.items():
        m = json.loads(json.dumps(base))
        for key, val in patch.items():
            if val is _DROP:
                m.pop(key, None)
            else:
                m[key] = val
        path = os.path.join(work, f"manifest_bad_{label}.json")
        with open(path, "w") as fh:
            json.dump(m, fh, indent=2)
        p = subprocess.run([sys.executable, SCORECARD, "--manifest", path, "--out-dir",
                            os.path.join(work, "out_bad")], cwd=_REPO, capture_output=True,
                           text=True)
        blob = (p.stdout or "") + (p.stderr or "")
        if p.returncode != 2:
            raise SystemExit(f"[smoke] FAIL (vi): {label} exited {p.returncode}, expected 2")
        if "Traceback" in blob:
            raise SystemExit(f"[smoke] FAIL (vi): {label} printed a traceback")
        if "REFUSED — bad manifest" not in blob or needle not in blob:
            raise SystemExit(f"[smoke] FAIL (vi): {label} message not actionable: {blob[-300:]}")
        print(f"[smoke] (vi) {label:<18} exit 2 — {blob.strip().splitlines()[-1][:110]}")
    truncated = os.path.join(work, "manifest_truncated.json")
    with open(truncated, "w") as fh:
        fh.write('{"schema": "imprint_scorecard_manifest/v1", "name": "x"')
    p = subprocess.run([sys.executable, SCORECARD, "--manifest", truncated, "--out-dir",
                        os.path.join(work, "out_bad")], cwd=_REPO, capture_output=True, text=True)
    blob = (p.stdout or "") + (p.stderr or "")
    if p.returncode != 2 or "Traceback" in blob:
        raise SystemExit(f"[smoke] FAIL (vi): truncated manifest exited {p.returncode}: "
                         f"{blob[-300:]}")
    print(f"[smoke] (vi) truncated_json  exit 2 — {blob.strip().splitlines()[-1][:110]}")
    print("[smoke] (vi) PASS")


def _sole_result(win_dir):
    found = [f for f in sorted(os.listdir(win_dir))
             if f.startswith("results_") and f.endswith(".json")]
    if len(found) != 1:
        raise SystemExit(f"[smoke] expected exactly one eval result in {win_dir}, got {found}")
    with open(os.path.join(win_dir, found[0])) as fh:
        return json.load(fh)


def leg_vii_minute_source(work, ckpt):
    """D30 rider (c) — the PLUMB direction: a minute-declared manifest reaches eval_vector with
    ``--panel-frame-source minute`` and the emitted artifact agrees. Negative control in the same
    leg: a daily-declared manifest emits NO source flag, so the 848 command line is unchanged."""
    print("\n" + "=" * 84)
    print("[smoke] LEG (vii) — minute-declared manifest ⇒ minute source in argv AND in the "
          "artifact (daily emits no flag)")
    print("=" * 84)
    # --- NEGATIVE CONTROL: the shipped DAILY path adds nothing to the command line -------------
    daily_m = sc_mod.load_manifest(os.path.join(work, "manifest_toysmoke.json"))
    daily_argv = sc_mod.build_argv(daily_m, daily_m["windows"][0], "/OUT", "TAG")
    if "--panel-frame-source" in daily_argv:
        raise SystemExit("[smoke] FAIL (vii): a DAILY-declared manifest emitted "
                         "--panel-frame-source — the 848 command line is no longer byte-identical")
    if daily_argv[daily_argv.index("--bar-minutes") + 1] != "1440":
        raise SystemExit("[smoke] FAIL (vii): the daily path no longer emits --bar-minutes 1440")
    print("[smoke] (vii) negative control: daily emits NO --panel-frame-source flag "
          "(--bar-minutes 1440, flag-gated default OFF)")

    manifest = write_inputs(work, ckpt, windows=MINUTE_WINDOWS, name="minutesrc",
                            panel_frame_source="minute", bar_minutes=MINUTE_BAR_MINUTES)
    m = sc_mod.load_manifest(manifest)
    argv = sc_mod.build_argv(m, m["windows"][0], "/OUT", "TAG")
    if "--panel-frame-source" not in argv:
        raise SystemExit("[smoke] FAIL (vii): a MINUTE-declared manifest emitted no source flag")
    i = argv.index("--panel-frame-source")
    if argv[i + 1] != "minute":
        raise SystemExit(f"[smoke] FAIL (vii): emitted source {argv[i + 1]!r}, expected 'minute'")
    if argv[i - 1] != os.path.abspath(m["frame_provenance"]):
        raise SystemExit("[smoke] FAIL (vii): the source flag does not follow the frame pin it "
                         "qualifies")
    if argv[argv.index("--bar-minutes") + 1] != str(MINUTE_BAR_MINUTES):
        raise SystemExit("[smoke] FAIL (vii): the manifest stride did not reach argv")
    # the ONLY difference from the daily command line for this same manifest is those two tokens
    probe = dict(m)
    probe["panel_frame_source"] = "daily"          # build_argv-only probe (not a loadable pair)
    if argv[:i] + argv[i + 2:] != sc_mod.build_argv(probe, m["windows"][0], "/OUT", "TAG"):
        raise SystemExit("[smoke] FAIL (vii): the minute argv differs from the daily argv by more "
                         "than --panel-frame-source minute")
    print(f"[smoke] (vii) argv: --panel-frame-source minute follows the frame pin; "
          f"--bar-minutes {MINUTE_BAR_MINUTES}; nothing else changed")

    out_dir = os.path.join(work, "out_minutesrc")
    p = run_scorecard(manifest, out_dir)
    if p.returncode != 0:
        print(p.stderr[-4000:])
        raise SystemExit("[smoke] FAIL (vii): the minute-sourced battery exited non-zero")
    if "--panel-frame-source minute" not in p.stdout:
        raise SystemExit("[smoke] FAIL (vii): the executed command line does not carry the flag")
    sc = json.loads(_read_bytes(os.path.join(out_dir, "scorecard_minutesrc.json")))
    sc_mod.validate_scorecard(sc)
    if sc["status"] != "OK" or sc["n_failed"]:
        raise SystemExit(f"[smoke] FAIL (vii): status={sc['status']} "
                         f"n_failed={sc['n_failed']} — {[r.get('reason') for r in sc['windows']]}")
    if sc["operator"]["panel_frame_source"] != "minute":
        raise SystemExit("[smoke] FAIL (vii): the header does not pin the minute source")
    if sc["operator"]["panel_frame_pin"]["bar_minutes"] != MINUTE_BAR_MINUTES \
            or sc["bar_minutes"] != MINUTE_BAR_MINUTES:
        raise SystemExit("[smoke] FAIL (vii): the header stride is not the manifest's")
    if "--panel-frame-source minute" not in sc["operator"]["argv_template"]:
        raise SystemExit("[smoke] FAIL (vii): argv_template hides the source")
    row = sc["windows"][0]
    if row["panel"]["panel_frame_source"] != "minute":
        raise SystemExit(f"[smoke] FAIL (vii): row observed "
                         f"{row['panel']['panel_frame_source']!r}, expected 'minute'")
    # …and the OBSERVATION is not an echo: eval_vector's OWN artifact says minute.
    res = _sole_result(os.path.join(out_dir, "windows", "min_a"))
    fb = res.get("panel_frame") or {}
    if fb.get("panel_frame_source") != "minute":
        raise SystemExit(f"[smoke] FAIL (vii): eval_vector's artifact says "
                         f"{fb.get('panel_frame_source')!r}, not 'minute'")
    if int(res.get("bar_minutes")) != MINUTE_BAR_MINUTES:
        raise SystemExit("[smoke] FAIL (vii): eval_vector graded a different stride")
    md = _read_bytes(os.path.join(out_dir, "scorecard_minutesrc.md")).decode()
    if "panel frame source" not in md or "**minute**" not in md:
        raise SystemExit("[smoke] FAIL (vii): the MD render does not disclose the price source")
    print(f"[smoke] (vii) PASS — eval artifact panel_frame_source='minute' "
          f"(n_slots {fb.get('n_slots')}, active {fb.get('n_active_slots')}, "
          f"stride {res.get('bar_minutes')}), header+row+MD all agree; "
          f"net@21 {row['tiers']['21']['sharpe']}")


def leg_viii_source_guards(work, ckpt):
    """D30 rider (c) — the REFUSAL directions, each shown firing WITH its negative control.
    Mutations are applied to REAL emitted results (legs (ii)/(vii)), through the very check the
    battery runs, so no guard here is provably-unreachable decoration."""
    print("\n" + "=" * 84)
    print("[smoke] LEG (viii) — declared/actual source mismatch REFUSES; absence-means-daily is "
          "read correctly; source/stride coherence")
    print("=" * 84)
    daily_m = sc_mod.load_manifest(os.path.join(work, "manifest_toysmoke.json"))
    minute_m = sc_mod.load_manifest(os.path.join(work, "manifest_minutesrc.json"))
    daily_w, minute_w = daily_m["windows"][0], minute_m["windows"][0]
    daily_res = _sole_result(os.path.join(work, "out", "windows", daily_w["name"]))
    minute_res = _sole_result(os.path.join(work, "out_minutesrc", "windows", minute_w["name"]))
    # The shipped 853 daily block carries NO panel_frame_source key (its absence IS 'daily'). If a
    # later task starts writing it explicitly that is fine — both spellings are accepted — but the
    # absence branch must keep working, so it is exercised SYNTHETICALLY below either way.
    daily_key = (daily_res.get("panel_frame") or {}).get("panel_frame_source", "<absent>")
    print(f"[smoke] (viii) the real DAILY result's panel_frame_source: {daily_key!r} "
          f"(absence IS 'daily' — the TASK_853 artifact convention)")

    def check(res, manifest, window):
        return sc_mod._check_result_provenance(json.loads(json.dumps(res)), manifest, window)

    def expect_ok(label, res, manifest, window):
        try:
            check(res, manifest, window)
        except sc_mod.WindowFailure as e:
            raise SystemExit(f"[smoke] FAIL (viii): negative control {label} was REFUSED: {e}")
        print(f"[smoke] (viii) NEGATIVE CONTROL ok   {label}")

    def expect_refused(label, res, manifest, window, needle):
        try:
            check(res, manifest, window)
        except sc_mod.WindowFailure as e:
            if needle not in str(e):
                raise SystemExit(f"[smoke] FAIL (viii): {label} refused for the wrong reason: {e}")
            print(f"[smoke] (viii) REFUSED             {label} — {str(e)[:120]}")
            return
        raise SystemExit(f"[smoke] FAIL (viii): {label} was ACCEPTED")

    def with_source(res, val):
        r = json.loads(json.dumps(res))
        if val is _DROP:
            r["panel_frame"].pop("panel_frame_source", None)
        else:
            r["panel_frame"]["panel_frame_source"] = val
        return r

    # 1. the negative controls: each declaration matched by its own real result, plus the
    #    absence branch asserted on the ACCEPT side (a key-less block IS a daily read)
    expect_ok("daily manifest + real daily result", daily_res, daily_m, daily_w)
    expect_ok("minute manifest + real minute result", minute_res, minute_m, minute_w)
    expect_ok("daily manifest + result with NO source key (absence == daily)",
              with_source(daily_res, _DROP), daily_m, daily_w)
    expect_ok("daily manifest + result saying 'daily' explicitly",
              with_source(daily_res, "daily"), daily_m, daily_w)
    # 2. absence is read as DAILY, not as a missing field — so a MINUTE declaration refuses it
    expect_refused("minute manifest + result with NO source key (absence == daily)",
                   with_source(minute_res, _DROP), minute_m, minute_w,
                   "panel_frame.panel_frame_source 'daily' != manifest 'minute'")
    # 3. both mismatch directions
    expect_refused("minute manifest + result says daily", with_source(minute_res, "daily"),
                   minute_m, minute_w, "!= manifest 'minute'")
    expect_refused("daily manifest + result says minute", with_source(daily_res, "minute"),
                   daily_m, daily_w, "!= manifest 'daily'")
    # 4. an unrecognised emitted source is not a source at all
    expect_refused("daily manifest + result says 'weekly'", with_source(daily_res, "weekly"),
                   daily_m, daily_w, "is not one of ['daily', 'minute']")
    # 5. a stride the battery did not ask for (the minute block records its own)
    strided = json.loads(json.dumps(minute_res))
    strided["panel_frame"]["bar_minutes"] = MINUTE_BAR_MINUTES * 2
    expect_refused("minute manifest + frame strided differently", strided, minute_m, minute_w,
                   "panel_frame.bar_minutes")

    # 6. the source/stride coherence rule, fired and not fired.
    # FIX PASS — the stride is now BOUNDED as well as coherence-checked: (minute, 1) used to be
    # accepted and would have graded the read of record on a ONE-MINUTE bar with status OK.
    lo, hi = sc_mod.MINUTE_BAR_MINUTES_RANGE
    for source, bars, needle in (("daily", 60, "INCOHERENT"), ("daily", 1, "INCOHERENT"),
                                 ("minute", 0, "INCOHERENT"), ("minute", -5, "INCOHERENT"),
                                 ("minute", 1, "OUT-OF-RANGE"),
                                 ("minute", lo - 1, "OUT-OF-RANGE"),
                                 ("minute", hi + 1, "OUT-OF-RANGE"),
                                 ("hourly", 1440, "expected one of")):
        msg = sc_mod.panel_frame_source_error(source, bars)
        if not msg or needle not in msg:
            raise SystemExit(f"[smoke] FAIL (viii): ({source}, {bars}) was accepted as coherent "
                             f"(msg={msg!r})")
        print(f"[smoke] (viii) coherence REFUSED   ({source}, {bars}) — {msg[:100]}")
    for source, bars in (("daily", 1440), ("minute", 1440), ("minute", 60),
                         ("minute", MINUTE_BAR_MINUTES)):
        msg = sc_mod.panel_frame_source_error(source, bars)
        if msg is not None:
            raise SystemExit(f"[smoke] FAIL (viii): coherent pair ({source}, {bars}) was refused: "
                             f"{msg}")
    print(f"[smoke] (viii) NEGATIVE CONTROL ok   coherent pairs (daily,1440) "
          f"(minute,{lo}..{hi})")
    # ... and the breadth-floor domain, the same way (D30 rider (b), fix pass).
    for bad in (-1, -0.0 - 1e-9, True, "0", 1e-300, 1.5, float("nan"), float("inf"), None):
        if sc_mod.require_active_slots_error(bad) is None:
            raise SystemExit(f"[smoke] FAIL (viii): require_active_slots {bad!r} was accepted")
    for ok in (0, 0.5, 1, 1.0, 58, 64.0):
        if sc_mod.require_active_slots_error(ok) is not None:
            raise SystemExit(f"[smoke] FAIL (viii): require_active_slots {ok!r} was refused: "
                             f"{sc_mod.require_active_slots_error(ok)}")
    print("[smoke] (viii) breadth-floor domain OK  (0 | (0,1] | integral >= 1; the rest refused)")

    # 7. the schema itself refuses a battery whose legs disagree about the table
    good = json.loads(_read_bytes(os.path.join(work, "out_minutesrc",
                                               "scorecard_minutesrc.json")))
    mixed = json.loads(json.dumps(good))
    mixed["windows"][0]["panel"]["panel_frame_source"] = "daily"
    try:
        sc_mod.validate_scorecard(mixed)
    except ValueError as e:
        print(f"[smoke] (viii) schema REFUSED      a row reading another table — {str(e)[:110]}")
    else:
        raise SystemExit("[smoke] FAIL (viii): the schema accepted a row whose source disagrees "
                         "with the battery's declaration")
    dropped = json.loads(json.dumps(good))
    dropped["windows"][0]["panel"].pop("panel_frame_source")
    try:
        sc_mod.validate_scorecard(dropped)
    except ValueError as e:
        print(f"[smoke] (viii) schema REFUSED      a row that does not say — {str(e)[:110]}")
    else:
        raise SystemExit("[smoke] FAIL (viii): the schema accepted a row with no source at all")
    # TASK_869 fix pass: the window_utc schema rule has the same teeth — a row whose window
    # resolution is MISSING or MALFORMED is refused (a pre-869 artifact, which carries no
    # window_utc, can therefore never produce an OK scorecard row: fail-closed, not null).
    no_wu = json.loads(json.dumps(good))
    no_wu["windows"][0].pop("window_utc", None)
    try:
        sc_mod.validate_scorecard(no_wu)
    except ValueError as e:
        print(f"[smoke] (viii) schema REFUSED      a row with no window_utc — {str(e)[:110]}")
    else:
        raise SystemExit("[smoke] FAIL (viii): the schema accepted a row with no window_utc")
    for bad_wu in (None, ["2025-09-14", "2025-12-12"], [1757808000], [1757808000, True],
                   [1765497600, 1757808000.5]):
        bad = json.loads(json.dumps(good))
        bad["windows"][0]["window_utc"] = bad_wu
        try:
            sc_mod.validate_scorecard(bad)
        except ValueError:
            continue
        raise SystemExit(f"[smoke] FAIL (viii): the schema accepted window_utc {bad_wu!r}")
    print("[smoke] (viii) schema REFUSED      malformed window_utc "
          "(null / date strings / wrong length / bool / non-int)")
    sc_mod.validate_scorecard(good)          # negative control: the unmutated artifact is valid
    print("[smoke] (viii) PASS")


def main():
    ap = argparse.ArgumentParser(description="TASK_862 scorecard fixture smoke")
    # ``/var/tmp``, NOT ``tempfile.gettempdir()``. On this workstation ``/tmp`` is a 27 GB tmpfs
    # (RAM-backed) and this battery's work dir is persistent by design — it keeps a built
    # checkpoint and every toy run's artifacts so re-runs are byte-comparable. MEASURED
    # 2026-07-31: that one directory grew to most of 27 GB and filled ``/tmp`` to 4 KB free,
    # at which point every unrelated process writing there (orbax saves, JAX spills, subprocess
    # stdout capture) failed with ENOSPC. ``/var/tmp`` is disk-backed and is the correct home for
    # scratch that is meant to SURVIVE between runs. Override with --work-dir; TMPDIR is
    # deliberately not consulted, so the default cannot silently land back on a tmpfs.
    ap.add_argument("--work-dir",
                    default=os.path.join("/var/tmp", "trader_862_scorecard_smoke"),
                    dest="work_dir",
                    help="scratch dir for the toy battery (fixed by default so re-runs are "
                         "byte-comparable); NOT inside the repo, and NOT on the /tmp tmpfs — it "
                         "is persistent scratch and filled a 27GB tmpfs once")
    ap.add_argument("--fresh", action="store_true",
                    help="delete the work dir first (forces a fresh checkpoint build)")
    args = ap.parse_args()

    work = os.path.abspath(args.work_dir)
    if args.fresh and os.path.isdir(work):
        shutil.rmtree(work)
    os.makedirs(work, exist_ok=True)
    print(f"[smoke] work dir: {work}")
    if not os.path.exists(os.path.join(_REPO, FIXTURE_DB_REL)):
        raise SystemExit(f"[smoke] fixture DB missing: {FIXTURE_DB_REL} "
                         f"(regenerate: fixtures/make_fixture.py)")

    ckpt = ensure_checkpoint(work)
    leg_ii(work, ckpt)
    leg_iii_missing_floor(work, ckpt)
    leg_iii_bad_frame(work, ckpt)
    leg_iv_malformed_floor(work, ckpt)
    leg_v_operator_observed(work, ckpt)
    leg_vi_manifest_refusals(work, ckpt)
    leg_vii_minute_source(work, ckpt)
    leg_viii_source_guards(work, ckpt)
    print("\n[smoke] TASK_862 SCORECARD SMOKE: PASS "
          "(legs ii, iii-a, iii-b, iv, v, vi, vii, viii)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
