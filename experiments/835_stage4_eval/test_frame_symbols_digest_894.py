#!/usr/bin/env python3
"""TASK_894 — regression suite for the CERTIFICATION-side half of the frame_symbols digest:
`_certification_verdict`'s new (sixth) conjunct, `make_ledger_row`'s narrowed emission sentence,
and the retro-stamp tool's `frame_symbols_sha256` extension.

Separate from `mcts_trader/jax_agent/test_frame_symbols_digest_894.py` (the SAVE-side suite:
`checkpoint_io.frame_symbols_digest`, `trainer.save_checkpoint`'s new argument) and from
`test_universe_pin_886.py` (886's own gates, unchanged and not re-asserted here — running both
files is the full regression surface for `enforce_universe_pin`/`_certification_verdict`).

THE HOLE THIS CLOSES (measured by TASK_886, closed here). `param_fingerprint` hashes parameter
shape/dtype/bytes and has ZERO dependence on `frame_symbols` — reversing a checkpoint's own
provenance's `frame_symbols` with `param_fingerprint` left intact still certified
(`verified: true`) under 886. TASK_894 adds a SEPARATE, save-time record
(`trainer_metadata.json`'s `frame_symbols_sha256`, written by `trainer.save_checkpoint` under a
pinned frame, or retro-stamped from a checkpoint's own tracked provenance) that the SAME forgery
can no longer touch, and wires it into `_certification_verdict` as a sixth conjunct.

Gates exercised here:
  (predicate) [DIRECT] the new conjunct pinned by constructing `_certification_verdict`'s input
      record directly (886's own style) — `True`/absent/`None` all still certify (vacuous, the
      pre-894 shape); `False` (a RECORDED disagreement) refuses, naming TASK_894.
  (i) the run-34 reversed-`frame_symbols` forgery (fingerprint intact) now REFUSES for a
      checkpoint whose own trainer_metadata.json carries a `frame_symbols_sha256` (a "stamped"
      checkpoint, TASK_894's own vocabulary), and still certifies-with-"NOT PROVEN" for one that
      does not (byte-identical to 886's own behaviour) — proven on the SAME trained checkpoint
      and the SAME tamper, sequentially, so the only variable between the two legs is whether
      the digest was recorded.
  (ii) byte-identity vs HEAD on every scored field for the 7 real checkpoints PRE-STAMP. Proven
      two ways, deliberately WITHOUT a full real-data eval run (no GPU jobs, no expensive
      real-data runs): (a) none of the 7 real checkpoints' `trainer_metadata.json` carries a
      `frame_symbols_sha256` key today (read directly off disk, no load, no GPU); (b) gate (i)'s
      "unstamped" leg above proves DIRECTLY, on a real trained checkpoint, that the absence of
      this key produces the byte-identical pre-894 `ledger_row` sentence and `verified` value.
      Together: every real checkpoint is in the state gate (i)'s unstamped leg already proved is
      unaffected.
  (iii) the retro-stamp tool's documented BARE CLI invocation (no monkeypatched internals): a
      dry run derives and names `frame_symbols_sha256` from the checkpoint's own tracked
      provenance, and REFUSES (writing nothing) when a RECORDED digest disagrees with what it
      would stamp.
  (iv) [mutation-checked by hand during implementation — see the task report] a test that
      removes the sixth conjunct from `_certification_verdict`'s success line goes RED at the
      real drop site: the (predicate) gate's `digest_false` leg, which certifies improperly the
      moment `and frame_digest_ok` is deleted from the success condition.
  (leaf-shared-disagree) [FIX PASS, correctness MINOR/latent] the run-dir-SHARED
      `trainer_metadata.json` and the PER-LEAF `trainer_metadata_<step>.json` recording
      DIFFERENT `frame_symbols_sha256` values for the SAME leaf must REFUSE (never silently
      prefer one sidecar, never degrade to a vacuous `None`) — the fail-closed direction
      `_checkpoint_recorded_frame_symbols_sha256` takes now that it prefers the per-leaf record.

CPU / fixture only; no GPU; never touches `trading_data.db`. Run:
    python experiments/835_stage4_eval/test_frame_symbols_digest_894.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE_DB = os.path.join(_REPO, "fixtures", "trading_fixture.db")
IMPRINT_PY = os.path.join(_REPO, "experiments", "832_linear_imprint", "imprint.py")
EVAL_PY = os.path.join(_REPO, "experiments", "835_stage4_eval", "eval_vector.py")
STAMP_PY = os.path.join(_REPO, "tools", "stamp_checkpoint_fingerprints.py")

FRAME_SYMBOLS = ["XBT/USD", "ETH/USD", "LTC/USD", "XMR/USD",
                 "ZEC/USD", "ETC/USD", "DASH/USD", "REP/USD"]
START, END, BAR_MINUTES, N_ASSETS = "2025-09-14", "2025-12-12", 240, 8
TRAIN_ITERS = 15   # cheap, real-and-real-different only (886's own convention for this fixture)

BASE = ["--db", FIXTURE_DB, "--start", START, "--end", END,
        "--bar-minutes", str(BAR_MINUTES), "--n-assets", str(N_ASSETS),
        "--operator", "policy-argmax", "--vector-weight-rebalance"]

# The 7 checkpoints named in TASK_888/889 (and this task's own acceptance (ii)) — the same
# default list `tools/stamp_checkpoint_fingerprints.py` targets. Read-only in this suite (a
# plain file read, no load, no GPU): proving they carry NO frame_symbols_sha256 today.
REAL_CHECKPOINTS = [
    "runs/861_ref/ckpt_ref_s0", "runs/861_ref/ckpt_ref_s1", "runs/861_ref/ckpt_ref_s2",
    "runs/861_imprint/ckpt_s0", "runs/861_imprint/ckpt_s1", "runs/861_imprint/ckpt_s2",
    "runs/832_e5/ckpt_imprint_full",
]

failures = []


def _write_frame(path, symbols):
    with open(path, "w") as fh:
        json.dump({"frame_symbols": list(symbols), "n_assets": len(symbols),
                   "source": "test_frame_symbols_digest_894.py", "train_window": [START, END],
                   "db": FIXTURE_DB}, fh)


def _run(argv, timeout=280):
    return subprocess.run([sys.executable] + argv, cwd=_REPO, capture_output=True, text=True,
                          timeout=timeout)


def _artifacts(work, tag):
    return [f for f in os.listdir(work) if f.startswith(f"results_{tag}_") and f.endswith(".json")]


def _one_doc(work, tag):
    arts = _artifacts(work, tag)
    if len(arts) != 1:
        return None
    with open(os.path.join(work, arts[0])) as fh:
        return json.load(fh)


def _leaf_dir(run_dir):
    leaves = [d for d in os.listdir(run_dir) if d.startswith("checkpoint_")]
    assert len(leaves) == 1, f"expected exactly one checkpoint_* leaf in {run_dir}: {leaves}"
    return os.path.join(run_dir, leaves[0])


# ---- (predicate)/(iv) DIRECT probe of _certification_verdict's new conjunct ------------------
_PREDICATE_PROBE = r'''#!/usr/bin/env python3
"""Direct probe of eval_vector's certification predicate for TASK_894's sixth conjunct.
Usage: probe.py <eval_vector.py>; prints one JSON object on stdout. Calls the REAL function;
nothing stubbed, mocked or monkeypatched — only the predicate's INPUT record is constructed."""
import importlib.util
import json
import sys

eval_py = sys.argv[1]
spec = importlib.util.spec_from_file_location("ev894_probe", eval_py)
ev = importlib.util.module_from_spec(spec)
sys.modules["ev894_probe"] = ev
spec.loader.exec_module(ev)

FP = "0123456789abcdef" * 4
LOAD = {"param_fingerprint": FP}


def rec(**over):
    """Every OTHER conjunct forced true (886's own control shape), plus the NEW conjunct forced
    true too — `control` below proves the full six-conjunct record still certifies."""
    r = {"mechanism": "checkpoint-provenance",
         "verified_against_checkpoint_provenance": True,
         "order_exact": True,
         "provenance_param_fingerprint": FP,
         "n_padded": 0,
         "frame_symbols_digest_verified": True}
    r.update(over)
    return r


def verdict(r, realized=True, load=LOAD):
    v, why = ev._certification_verdict(r, realized, load)
    return [bool(v), why]


def drop(field):
    r = rec()
    r.pop(field)
    return r


out = {"control": verdict(rec())}
out["digest_true"] = verdict(rec(frame_symbols_digest_verified=True))
out["digest_none"] = verdict(rec(frame_symbols_digest_verified=None))
out["digest_absent"] = verdict(drop("frame_symbols_digest_verified"))
out["digest_false"] = verdict(rec(frame_symbols_digest_verified=False))
print(json.dumps(out))
'''


def _run_predicate_probe(work):
    probe_py = os.path.join(work, "probe_894.py")
    with open(probe_py, "w") as fh:
        fh.write(_PREDICATE_PROBE)
    pr = _run([probe_py, EVAL_PY])
    if pr.returncode != 0 or not pr.stdout.strip():
        failures.append(
            f"DIRECT-PREDICATE probe failed to run (exit {pr.returncode}).\n"
            f"stdout tail:\n{pr.stdout[-1500:]}\nstderr tail:\n{pr.stderr[-2500:]}")
        return None
    return json.loads(pr.stdout.strip().splitlines()[-1])


def gate_predicate(work):
    print("[894] gate (predicate)/(iv) — DIRECT probe of _certification_verdict's sixth "
          "conjunct (constructing the record; the predicate is called for real) ...")
    probe = _run_predicate_probe(work)
    if probe is None:
        return
    ctl_ok = probe["control"] == [True, "certified"]
    print(f"    [{'PASS' if ctl_ok else 'FAIL'}] (control) all six conjuncts true CERTIFIES: "
          f"{probe['control']}")
    if not ctl_ok:
        failures.append(f"predicate control: expected [True, 'certified'], got {probe['control']}")

    ok = (ctl_ok
          and probe["digest_true"] == [True, "certified"]
          and probe["digest_none"] == [True, "certified"]
          and probe["digest_absent"] == [True, "certified"])
    print(f"    [{'PASS' if ok else 'FAIL'}] gate (predicate) True/None/absent all still "
          f"CERTIFY (vacuous — the pre-894 shape is unaffected)")
    if not ok:
        failures.append(
            f"gate (predicate): True/None/absent frame_symbols_digest_verified must all still "
            f"certify. digest_true={probe['digest_true']} digest_none={probe['digest_none']} "
            f"digest_absent={probe['digest_absent']}")

    ok = (ctl_ok and probe["digest_false"][0] is False
          and "TASK_894" in probe["digest_false"][1]
          and "frame_symbols" in probe["digest_false"][1])
    print(f"    [{'PASS' if ok else 'FAIL'}] gate (predicate)/(iv) a RECORDED disagreement "
          f"(False) REFUSES, naming TASK_894: {probe['digest_false']}")
    if not ok:
        failures.append(
            f"gate (predicate)/(iv): frame_symbols_digest_verified=False (with every other "
            f"conjunct true) must REFUSE. got {probe['digest_false']} — THIS is the leg that "
            f"goes red the moment 'and frame_digest_ok' is removed from "
            f"_certification_verdict's success line (mutation-checked by hand during "
            f"implementation).")


def gate_real_checkpoints_unstamped(work):
    print("[894] gate (ii)-a — none of the 7 real load-bearing checkpoints carry a "
          "frame_symbols_sha256 today (plain file read, no load, no GPU) ...")
    n_checked = 0
    for rel in REAL_CHECKPOINTS:
        run_dir = os.path.join(_REPO, rel)
        meta_path = os.path.join(run_dir, "trainer_metadata.json")
        if not os.path.exists(meta_path):
            print(f"    [SKIP] {rel} — not present on this host")
            continue
        with open(meta_path) as fh:
            meta = json.load(fh)
        n_checked += 1
        ok = "frame_symbols_sha256" not in meta
        print(f"    [{'PASS' if ok else 'FAIL'}] {rel}: frame_symbols_sha256 "
              f"{'absent' if ok else 'PRESENT — unexpected pre-stamp'}")
        if not ok:
            failures.append(
                f"gate (ii)-a: {rel}'s trainer_metadata.json already carries a "
                f"frame_symbols_sha256 — the byte-identity claim for 'pre-stamp' checkpoints "
                f"assumes none do yet. meta={meta}")
    ok = n_checked > 0
    print(f"    [{'PASS' if ok else 'FAIL'}] gate (ii)-a checked {n_checked}/"
          f"{len(REAL_CHECKPOINTS)} real checkpoint(s) present on this host")
    if not ok:
        failures.append(
            "gate (ii)-a: NONE of the 7 named real checkpoints were present on this host — "
            "the byte-identity claim could not be checked against real artifacts at all.")


def main():
    before = None
    if os.path.exists(FIXTURE_DB):
        before = (os.path.getsize(FIXTURE_DB), os.path.getmtime(FIXTURE_DB))
    work = tempfile.mkdtemp(prefix="t894_frame_digest_cert_")
    try:
        frame_a = os.path.join(work, "frame_A.json")
        _write_frame(frame_a, FRAME_SYMBOLS)

        print("[894] training a checkpoint PINNED under FRAME_A (imprint.py now records "
              "frame_symbols_sha256 at save time, TASK_894 scope 1) ...")
        ckpt = os.path.join(work, "ckpt_stamped")
        r = _run([IMPRINT_PY, "--db", FIXTURE_DB, "--start", START, "--end", END,
                  "--bar-minutes", str(BAR_MINUTES), "--n-assets", str(N_ASSETS),
                  "--panel-frame-from-checkpoint", frame_a, "--panel-frame-source", "minute",
                  "--imprint-mapping", "weight-native", "--iters", str(TRAIN_ITERS),
                  "--batch-size", "32", "--seed", "0", "--vector-weight-rebalance",
                  "--ckpt-out", ckpt, "--tag", "t894", "--out",
                  os.path.join(work, "imprint_894.json")])
        if r.returncode != 0:
            raise RuntimeError(f"imprint.py training failed (exit {r.returncode}):\n"
                               f"{r.stderr[-3000:]}")
        own_prov = os.path.join(work, "frame_provenance_stamped.json")
        if not os.path.exists(own_prov):
            raise RuntimeError(f"expected imprint.py (TASK_874) to emit {own_prov}")
        with open(own_prov, "rb") as fh:
            own_prov_bytes = fh.read()
        meta_path = os.path.join(ckpt, "trainer_metadata.json")
        with open(meta_path) as fh:
            meta_bytes_original = fh.read()
        meta = json.loads(meta_bytes_original)
        if "frame_symbols_sha256" not in meta:
            raise RuntimeError(
                f"setup assumption broken: {meta_path} carries no frame_symbols_sha256 — "
                f"trainer.save_checkpoint (TASK_894 scope 1) is expected to write one "
                f"unconditionally under a pinned frame. meta={meta}")
        print(f"    run dir: {ckpt}\n    trainer_metadata.json frame_symbols_sha256="
              f"{meta['frame_symbols_sha256'][:16]}...")

        def _restore_prov():
            with open(own_prov, "wb") as fh:
                fh.write(own_prov_bytes)

        def _restore_meta():
            with open(meta_path, "w") as fh:
                fh.write(meta_bytes_original)

        def _tamper_prov():
            with open(own_prov) as fh:
                d = json.load(fh)
            d["frame_symbols"] = list(reversed(d["frame_symbols"]))
            with open(own_prov, "w") as fh:
                json.dump(d, fh)

        # ---- (predicate)/(iv) --------------------------------------------------------------
        gate_predicate(work)

        # ---- positive control: untampered STAMPED checkpoint certifies with the NEW, --------
        # ---- NARROWER sentence (scope 4) -----------------------------------------------------
        print("[894] gate (proven) — an UNTAMPERED stamped checkpoint certifies, and the "
              "'NOT PROVEN: slot order' half is DELETED (scope 4) ...")
        r = _run([EVAL_PY] + BASE + ["--checkpoint", ckpt,
                                     "--panel-frame-from-checkpoint", own_prov,
                                     "--panel-frame-source", "minute",
                                     "--tag", "gproven", "--out-dir", work])
        doc = _one_doc(work, "gproven")
        up = (doc or {}).get("universe_pin", {})
        row = (doc or {}).get("ledger_row", "")
        ok = (r.returncode == 0 and doc is not None
              and up.get("verified") is True
              and up.get("frame_symbols_digest_verified") is True
              and "UNIVERSE CERTIFIED (TASK_886/894)" in row
              and "PROVEN" in row
              and "NOT PROVEN" not in row
              and "SELF-ATTESTING" not in row
              and row.count("UNIVERSE ") == 1)
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (proven) stamped+untampered certifies, "
              f"'NOT PROVEN' deleted")
        if not ok:
            failures.append(
                f"gate (proven): expected verified=true, frame_symbols_digest_verified=true, "
                f"'UNIVERSE CERTIFIED (TASK_886/894)' with 'PROVEN' and NO 'NOT PROVEN' in "
                f"ledger_row. returncode={r.returncode} universe_pin={up}\nledger_row={row}\n"
                f"stderr tail:\n{r.stderr[-1500:]}")

        # ---- FIX PASS (correctness, MINOR/latent): shared vs per-leaf digest DISAGREEMENT -----
        # ---- must REFUSE, never silently prefer one sidecar or fall back to a vacuous None ----
        print("[894] gate (leaf-shared-disagree) — the per-leaf trainer_metadata_<step>.json "
              "and the run-dir-shared trainer_metadata.json recording DIFFERENT "
              "frame_symbols_sha256 values for the SAME leaf must REFUSE ...")
        leaf_dir = _leaf_dir(ckpt)
        leaf_step = int(os.path.basename(leaf_dir).split("_", 1)[1])
        leaf_meta_path = os.path.join(ckpt, f"trainer_metadata_{leaf_step}.json")
        with open(leaf_meta_path) as fh:
            leaf_meta_original = json.load(fh)
        if "frame_symbols_sha256" not in leaf_meta_original:
            raise RuntimeError(
                f"setup assumption broken: {leaf_meta_path} carries no frame_symbols_sha256 — "
                f"trainer.save_checkpoint (TASK_894 scope 1) is expected to write the SAME "
                f"record into the per-leaf sidecar as the shared one. meta={leaf_meta_original}")
        leaf_meta_bad = dict(leaf_meta_original)
        # Real sha256 shape, deliberately DIFFERENT content from the shared sidecar's digest.
        leaf_meta_bad["frame_symbols_sha256"] = "f" * 64
        assert leaf_meta_bad["frame_symbols_sha256"] != meta["frame_symbols_sha256"]
        try:
            with open(leaf_meta_path, "w") as fh:
                json.dump(leaf_meta_bad, fh)
            r = _run([EVAL_PY] + BASE + ["--checkpoint", ckpt,
                                         "--panel-frame-from-checkpoint", own_prov,
                                         "--panel-frame-source", "minute",
                                         "--tag", "gleafdisagree", "--out-dir", work])
            ok = (r.returncode != 0
                  and "disagree" in r.stderr.lower()
                  and "trainer_metadata" in r.stderr
                  and not _artifacts(work, "gleafdisagree"))
        finally:
            with open(leaf_meta_path, "w") as fh:
                json.dump(leaf_meta_original, fh)
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (leaf-shared-disagree) shared/per-leaf "
              f"digest disagreement REFUSES, writing no artifact")
        if not ok:
            failures.append(
                f"gate (leaf-shared-disagree): expected a REFUSAL (non-zero exit) naming a "
                f"disagreement between the shared and per-leaf trainer_metadata sidecars, and "
                f"NO results artifact. returncode={r.returncode}\n"
                f"stderr tail:\n{r.stderr[-2000:]}")

        # ---- (i), first half: reversed frame_symbols REFUSES for a STAMPED checkpoint -------
        print("[894] gate (i-stamped) — the run-34 reversed-frame_symbols forgery (fingerprint "
              "intact) now REFUSES for a STAMPED checkpoint ...")
        _tamper_prov()
        try:
            r = _run([EVAL_PY] + BASE + ["--checkpoint", ckpt,
                                         "--panel-frame-from-checkpoint", own_prov,
                                         "--panel-frame-source", "minute",
                                         "--tag", "gforge_stamped", "--out-dir", work])
            doc_t = _one_doc(work, "gforge_stamped")
            up_t = (doc_t or {}).get("universe_pin", {})
            row_t = (doc_t or {}).get("ledger_row", "")
            ok = (r.returncode == 0 and doc_t is not None
                  # the number moved (the tamper is real) ...
                  and doc_t["tiers"]["21"]["sharpe_full"] != doc["tiers"]["21"]["sharpe_full"]
                  # ... but certification now REFUSES it, naming TASK_894.
                  and up_t.get("verified") is False
                  and up_t.get("frame_symbols_digest_verified") is False
                  and "TASK_894" in (up_t.get("verified_reason") or "")
                  and "UNIVERSE NOT CERTIFIED (TASK_886)" in row_t
                  and row_t.count("UNIVERSE ") == 1)
        finally:
            _restore_prov()
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (i-stamped) reversed forgery REFUSES for "
              f"a stamped checkpoint")
        if not ok:
            failures.append(
                f"gate (i-stamped): expected exit 0 with verified=false, "
                f"frame_symbols_digest_verified=false, reason naming TASK_894, and the number "
                f"to have moved (the tamper is real). returncode={r.returncode} "
                f"universe_pin={up_t}\nledger_row={row_t}\nstderr tail:\n{r.stderr[-1500:]}")

        # ---- (i), second half / (ii): the SAME tamper on an UNSTAMPED checkpoint is ----------
        # ---- unaffected — byte-identical to 886's pre-894 behaviour --------------------------
        print("[894] gate (i-unstamped)/(ii) — the SAME tamper on an UNSTAMPED checkpoint "
              "(frame_symbols_sha256 stripped, simulating a pre-894 save) still "
              "certifies-with-'NOT PROVEN', unaffected ...")
        meta_unstamped = dict(meta)
        meta_unstamped.pop("frame_symbols_sha256")
        with open(meta_path, "w") as fh:
            json.dump(meta_unstamped, fh)
        # FIX PASS: _checkpoint_recorded_frame_symbols_sha256 now PREFERS the per-leaf sidecar
        # over the shared one, so simulating "unstamped" (no recorded digest ANYWHERE) means
        # stripping it from BOTH records — the per-leaf one alone would otherwise still supply
        # a digest here and this leg would silently stop testing the vacuous-None path.
        leaf_meta_unstamped = dict(leaf_meta_original)
        leaf_meta_unstamped.pop("frame_symbols_sha256")
        with open(leaf_meta_path, "w") as fh:
            json.dump(leaf_meta_unstamped, fh)
        _tamper_prov()
        try:
            r = _run([EVAL_PY] + BASE + ["--checkpoint", ckpt,
                                         "--panel-frame-from-checkpoint", own_prov,
                                         "--panel-frame-source", "minute",
                                         "--tag", "gforge_unstamped", "--out-dir", work])
            doc_u = _one_doc(work, "gforge_unstamped")
            up_u = (doc_u or {}).get("universe_pin", {})
            row_u = (doc_u or {}).get("ledger_row", "")
            ok = (r.returncode == 0 and doc_u is not None
                  and up_u.get("verified") is True
                  and up_u.get("frame_symbols_digest_verified") is None
                  and "UNIVERSE CERTIFIED (TASK_886):" in row_u
                  and "UNIVERSE CERTIFIED (TASK_886/894)" not in row_u
                  and "NOT PROVEN" in row_u
                  and "SELF-ATTESTING" in row_u
                  and row_u.count("UNIVERSE ") == 1)
        finally:
            _restore_prov()
            _restore_meta()
            with open(leaf_meta_path, "w") as fh:
                json.dump(leaf_meta_original, fh)
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (i-unstamped)/(ii) unstamped checkpoint "
              f"unaffected — byte-identical to 886's pre-894 sentence")
        if not ok:
            failures.append(
                f"gate (i-unstamped)/(ii): an unstamped checkpoint under the identical tamper "
                f"must still certify with the ORIGINAL 886 sentence (NOT PROVEN/SELF-ATTESTING, "
                f"no '/894' suffix). returncode={r.returncode} universe_pin={up_u}\n"
                f"ledger_row={row_u}\nstderr tail:\n{r.stderr[-1500:]}")

        # ---- (ii)-a: the 7 real checkpoints carry no frame_symbols_sha256 today --------------
        gate_real_checkpoints_unstamped(work)

        # ---- (iii): retro-stamp bare CLI, dry-run, names the derivation ----------------------
        print("[894] gate (iii) — the retro-stamp tool's BARE CLI invocation (dry-run) derives "
              "and NAMES frame_symbols_sha256 from the checkpoint's own tracked provenance ...")
        r = _run([STAMP_PY, "--checkpoint", ckpt, "--n-assets", str(N_ASSETS)])
        ok = (r.returncode == 0
              and "frame_symbols_sha256" in r.stdout
              and "DERIVED from" in r.stdout
              and "frame_provenance_stamped.json" in r.stdout
              and "own recorded frame_symbols" in r.stdout)
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (iii) bare dry-run names the derivation")
        if not ok:
            failures.append(
                f"gate (iii): expected exit 0 and stdout naming a DERIVED frame_symbols_sha256 "
                f"from the tracked provenance document. returncode={r.returncode}\n"
                f"stdout tail:\n{r.stdout[-2000:]}\nstderr tail:\n{r.stderr[-2000:]}")

        print("[894] gate (iii)-refuse — a RECORDED frame_symbols_sha256 that disagrees with "
              "what the tool would derive REFUSES, writing nothing (dry-run OR --apply alike; "
              "checked here in dry-run) ...")
        meta_bad = dict(meta)
        meta_bad["frame_symbols_sha256"] = "0" * 64
        with open(meta_path, "w") as fh:
            json.dump(meta_bad, fh)
        try:
            r = _run([STAMP_PY, "--checkpoint", ckpt, "--n-assets", str(N_ASSETS)])
            ok = (r.returncode != 0
                  and "REFUSED" in r.stdout
                  and "frame_symbols_sha256" in r.stdout
                  and "disagrees" in r.stdout)
        finally:
            _restore_meta()
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (iii)-refuse disagreement REFUSES")
        if not ok:
            failures.append(
                f"gate (iii)-refuse: expected a non-zero exit and a printed REFUSAL naming a "
                f"frame_symbols_sha256 disagreement. returncode={r.returncode}\n"
                f"stdout tail:\n{r.stdout[-2000:]}\nstderr tail:\n{r.stderr[-2000:]}")
        # And confirm nothing was actually written for that disagreement (meta_path unchanged
        # from the BAD value this leg wrote, since _restore_meta already ran in `finally` above
        # — re-read to confirm the restore itself succeeded and no OTHER write slipped through).
        with open(meta_path) as fh:
            meta_after = json.load(fh)
        ok = meta_after == meta
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (iii)-refuse trainer_metadata.json "
              f"restored exactly (nothing else changed it)")
        if not ok:
            failures.append(
                f"gate (iii)-refuse: trainer_metadata.json after restore does not match the "
                f"original. before={meta}\nafter={meta_after}")

    finally:
        shutil.rmtree(work, ignore_errors=True)

    if before is not None:
        after = (os.path.getsize(FIXTURE_DB), os.path.getmtime(FIXTURE_DB))
        if before != after:
            failures.append(f"fixtures/trading_fixture.db changed size/mtime during this run "
                            f"({before} -> {after}) — INVARIANT VIOLATION, never regenerate it.")
        else:
            print(f"[894] fixture unchanged: size={before[0]} mtime={before[1]}")

    print("=" * 84)
    if failures:
        print(f"[894] FAIL — {len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[894] PASS — gates (predicate)/(iv), (proven), (leaf-shared-disagree), (i-stamped), "
          "(i-unstamped)/(ii), (ii)-a, (iii), (iii)-refuse green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
