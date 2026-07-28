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
        name, start >= end, truncated JSON) exit 2 with a clean message, never a traceback.

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
import tempfile

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
                 floor_overrides=None):
    """Write provenance + toy floor artifacts + manifest. Returns the manifest path.

    ``floor_overrides`` maps a window name to the literal ``floor`` object to write (used by
    leg (iv) to plant a non-numeric / explicitly-null floor)."""
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
    with open(manifest_path, "w") as fh:
        json.dump({"schema": sc_mod.MANIFEST_SCHEMA,
                   "name": name,
                   "checkpoint": ckpt,
                   "frame_provenance": prov_path,
                   "db": FIXTURE_DB_REL,
                   "n_assets": N_ASSETS,
                   "eval_seed": 0,
                   "notes": "TASK_862 fixture smoke (toy battery — no thresholds applied)",
                   "windows": win_specs}, fh, indent=2)
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


BAD_MANIFESTS = {
    "unknown_key": ({"eval_seeds": 3}, "unknown key"),
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
        m.update(patch)
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
        print(f"[smoke] (vi) {label:<16} exit 2 — {blob.strip().splitlines()[-1][:110]}")
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


def main():
    ap = argparse.ArgumentParser(description="TASK_862 scorecard fixture smoke")
    ap.add_argument("--work-dir", default=os.path.join(tempfile.gettempdir(),
                                                       "trader_862_scorecard_smoke"),
                    dest="work_dir",
                    help="scratch dir for the toy battery (fixed by default so re-runs are "
                         "byte-comparable); NOT inside the repo")
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
    print("\n[smoke] TASK_862 SCORECARD SMOKE: PASS (legs ii, iii-a, iii-b, iv, v, vi)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
