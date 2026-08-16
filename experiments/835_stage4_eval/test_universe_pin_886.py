#!/usr/bin/env python3
"""TASK_886 — regression test for eval_vector.py's universe-pin CERTIFICATION gates.

Separate file from `test_universe_pin_883.py` (deliberate): 883R's suite pins its own gates
(i)-(ix) — the refuse/disclaim machinery. This file adds 886's NEW gates on top without
re-asserting 883R's own coverage, so the two suites can evolve independently and a reader
always knows which task a failing gate belongs to. 886's implementation is strictly ADDITIVE
onto 883R's `universe_pin` record (new keys: `mechanism`, `verified`, `verified_reason`,
`realized_matches_requested`, `n_slots`, `n_padded`, `provenance_path`, `sha256`,
`provenance_param_fingerprint`) — running BOTH files is the full regression surface.

Also separate from `mcts_trader/jax_agent/test_checkpoint_io.py` (TASK_888/889's own suite,
which owns the checkpoint LOAD PATH: resolution, hang-safety, object-identity, foreign-shape
refusal, architecture binding). This row does not rebuild or re-test any of that — it consumes
`checkpoint_io`'s resolver/loader as a black box and tests only what THIS row adds: the
certification predicate, the provenance-sibling resolution (Hole 1), and the ledger wording.

Context (D36 ruling out of TASK_883's halt; D38 re-cut this row to CONTRACT ONLY; D39's
retro-stamp + 888R/889 discharged the premise that made `verified: true` unreachable). 883R
made the tool unable to LIE (it can only refuse or disclaim); this row makes it able to
CERTIFY — and only against evidence.

WHAT "CERTIFIED" MEANS HERE. `verified: true` requires FOUR things:
  1. the checkpoint's OWN provenance was found (from the RESOLVED checkpoint dir, so every
     spelling reaches the same verdict) and agreed order-exact with the requested universe;
  2. the panel `build_panel` actually RETURNED equals what was requested, order-exact;
  3. that provenance records a `param_fingerprint` EQUAL to the fingerprint of the parameters
     actually LOADED;
  4. the panel carries zero padded slots (`n_padded == 0`).
Without (3), certification was two JSON files agreeing with each other — both reachable from a
path derived from the checkpoint DIRECTORY'S NAME, with nothing ever compared against the
weights. `imprint.py` (TASK_888/889) now writes `param_fingerprint` at SAVE time for every
checkpoint it emits, and the retro-stamp (`tools/stamp_checkpoint_fingerprints.py`) backfilled
it into the tracked provenance documents for the surviving load-bearing checkpoints — so
`verified: true` is REACHABLE today, not merely a predicate that could someday fire. Gates
(3b-)/(3b+) below prove BOTH directions on a real, freshly-trained checkpoint.

Gates exercised here:
  (0c) the fixture caveat in ledger_row is keyed on the DB actually opened, not on --tag — and
       survives a SYMLINK spelling of the fixture path (realpath, not abspath) AND a
       byte-identical COPY of the fixture at an unrelated path (CONTENT identity: 0c-3/0c-4),
       while a 24GiB file is rejected on SIZE without being read (trading_data.db is never
       hashed).
  (mech)/(order)/(conj) [DIRECT] every conjunct of `_certification_verdict` is pinned by
       CONSTRUCTING the record and calling the predicate, not by routing through
       `enforce_universe_pin`. A conjunct that holds only because its producer happens to
       hardcode a companion field in the same dict literal is not enforced by the predicate —
       measured: a hand-built `{"mechanism": "none", ...}` record with the other conjuncts true
       used to return `(True, "certified")`, i.e. an UNPINNED run certifying, and deleting the
       `mechanism == "none"` branch changed no test outcome in either suite.
  (hist) [DIRECT] a historical pre-886 `universe_pin` (no `verified`/`verified_reason` keys)
       renders a named state in `ledger_row`, never the literal "NOT certified — None".
  (order-tamper-a/b)/(pin-ok)/(pin-ambig) THE LIMIT OF THE FINGERPRINT. `param_fingerprint`
       hashes parameter shape/dtype/bytes per leaf and has ZERO dependence on `frame_symbols`
       — weights do not encode symbol names. So certification proves WHICH WEIGHTS loaded and
       NOT the symbol ORDER they were trained under: measured on the fixture, reversing
       `frame_symbols` in the checkpoint's own provenance with the fingerprint left intact
       still certifies and moves net@21 (the per-slot weights score against different real
       assets). Binding the order properly means writing a `frame_symbols` digest INTO the
       checkpoint at SAVE time — TASK_888/889's surface, a new row, explicitly out of scope
       here. What this row owes is (1) a CERTIFIED sentence that states what is proven and
       names the self-attesting half instead of claiming it, and (2) the one tamper check that
       IS available: D39's committed, co-located `*.sha256` pins (`runs/861_ref/`,
       `runs/832_e5/`) — verified where present, REFUSED on mismatch or on two pins
       disagreeing, and recorded as absent where there is none. Read-only: this program never
       creates or refreshes a `.sha256`.
  (pin-sync)/(pin-fifo) THE LIMIT OF THE PIN, and the pin lookup's own read safety. The pin is
       a SIBLING FILE behind the same write permission as the document, so it catches only an
       ASYMMETRIC change (document moved, pin left stale — corruption, a partial sync, a stale
       copy, a hand-edit that forgot the pin) and NOT a writer who updates both: (pin-sync)
       reverses `frame_symbols` AND recomputes the pin, and pins that the run still certifies
       while the CERTIFIED sentence states the check's actual reach instead of claiming
       tampering is "ruled out". (pin-fifo) covers the lookup rule's cost: it opens every
       `*.sha256` in the directory before knowing which one names the document, so a FIFO under
       any name there must be SKIPPED unopened — asserted under its own fail-fast timeout,
       because the failure mode is a hang, not a wrong answer.
  (i)  both committed --checkpoint spellings of the same checkpoint (run dir / orbax leaf) AND
       a SYMLINK to it produce IDENTICAL universe_pin output (Hole 1: `_own_provenance_path_for`
       now resolves the checkpoint's RUN DIR first, so every spelling reaches the same
       provenance-sibling path), and a REVERSED pin refuses under every spelling.
  (3a) a provenance document that NAMES A DIFFERENT CHECKPOINT refuses (the documented
       non-injective `ckpt_foo`/`foo` sibling mapping is enough to reach this by accident).
  (3b) certification is bound to CONTENT: a provenance with no fingerprint => verified:false
       with an explicit machine-readable reason; a provenance whose fingerprint CONTRADICTS the
       loaded parameters REFUSES; a provenance whose fingerprint MATCHES (the ordinary,
       unmutated case for a freshly-trained checkpoint) CERTIFIES.
  (echo)/(nul) a refusal may name counts, paths and indices but never file content: an enormous
       recorded "checkpoint" value is CLIPPED, and a NUL byte in it (valid JSON) produces a
       NAMED refusal instead of a bare ValueError out of os.path.realpath.
  (ii) [regression] a --symbols list with a DB-absent symbol still REFUSES (883R's realized-
       panel assertion), so universe_pin.verified can never read true for it.
  (iii)/(iv) universe_pin.mechanism is present on EVERY artifact (including "none" and the NEW
       "periods-file" split), and ledger_row names it inline in every case — as ONE coherent
       statement per state.

CPU / fixture only; no GPU; never touches `trading_data.db`. Run:
    python experiments/835_stage4_eval/test_universe_pin_886.py
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE_DB = os.path.join(_REPO, "fixtures", "trading_fixture.db")
IMPRINT_PY = os.path.join(_REPO, "experiments", "832_linear_imprint", "imprint.py")
EVAL_PY = os.path.join(_REPO, "experiments", "835_stage4_eval", "eval_vector.py")

# The fixture's own 8-symbol minute-slice universe (eval_vector.SMOKE_SYMBOLS / 883R's /
# 882's FRAME_SYMBOLS) — reused, not re-derived.
FRAME_SYMBOLS = ["XBT/USD", "ETH/USD", "LTC/USD", "XMR/USD",
                 "ZEC/USD", "ETC/USD", "DASH/USD", "REP/USD"]
START, END, BAR_MINUTES, N_ASSETS = "2025-09-14", "2025-12-12", 240, 8
TRAIN_ITERS = 15   # cheap: these checks never depend on the checkpoint being GOOD, only real
                   # and REAL-DIFFERENT between the two trained checkpoints (0b's fingerprint
                   # inequality leg trains a second one under a different --seed).

BASE = ["--db", FIXTURE_DB, "--start", START, "--end", END,
        "--bar-minutes", str(BAR_MINUTES), "--n-assets", str(N_ASSETS),
        "--operator", "policy-argmax", "--vector-weight-rebalance"]


# 886 fix pass — the DIRECT-PREDICATE probe. Written into the temp work dir and run as its own
# interpreter (the driver stays free of jax state). It CALLS THE REAL FUNCTIONS: nothing is
# stubbed, mocked or monkeypatched. The only thing it constructs is the predicate's INPUT
# record — which is the entire point. Every gate above reaches `_certification_verdict` through
# `enforce_universe_pin`, so a conjunct that is enforced only by what THAT function happens to
# put in the same dict literal is invisible to them: the judge measured that deleting the
# `mechanism == "none"` branch changed no outcome in either suite while a hand-built
# `{"mechanism": "none", ...}` record with the other conjuncts true returned
# `(True, "certified")` — an UNPINNED run certifying. These checks pin the predicate itself.
_PREDICATE_PROBE = r'''#!/usr/bin/env python3
"""Direct probe of eval_vector's certification predicate, fixture identity and ledger wording.
Usage: probe.py <eval_vector.py> <config.json>; prints one JSON object on stdout."""
import importlib.util
import json
import os
import sys
import time

eval_py, cfg_path = sys.argv[1], sys.argv[2]
with open(cfg_path) as fh:
    cfg = json.load(fh)

spec = importlib.util.spec_from_file_location("ev886_probe", eval_py)
ev = importlib.util.module_from_spec(spec)
sys.modules["ev886_probe"] = ev
spec.loader.exec_module(ev)

FP = "0123456789abcdef" * 4
LOAD = {"param_fingerprint": FP}


def rec(**over):
    """A record with EVERY conjunct of _certification_verdict forced TRUE. `control` below
    proves it DOES certify, so any False outcome under a single-field override is attributable
    to that one field — the mutation is real, not vacuous."""
    r = {"mechanism": "checkpoint-provenance",
         "verified_against_checkpoint_provenance": True,
         "order_exact": True,
         "provenance_param_fingerprint": FP,
         "n_padded": 0}
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
# --- MAJOR 1: `mechanism` is a LIVE conjunct, not a reason-string lookup --------------------
out["mech_none"] = verdict(rec(mechanism="none"))
out["mech_absent"] = verdict(drop("mechanism"))
out["mech_null"] = verdict(rec(mechanism=None))
out["mech_unknown"] = verdict(rec(mechanism="wishful-thinking"))
out["mech_periods"] = verdict(rec(mechanism="periods-file"))
out["mech_explicit"] = verdict(rec(mechanism="explicit-symbols"))
# --- MAJOR 1 (audit of the other conjuncts): `order_exact` is a LIVE conjunct too -----------
# `verified_against_checkpoint_provenance` records only that a provenance was FOUND; the
# AGREEMENT half of conjunct (1) lived solely in `_cross_check_own_provenance` RAISING.
out["order_false"] = verdict(rec(order_exact=False))
out["order_null"] = verdict(rec(order_exact=None))
out["order_absent"] = verdict(drop("order_exact"))
# --- the conjuncts that were already live: control that this probe can see them fail --------
out["prov_false"] = verdict(rec(verified_against_checkpoint_provenance=False))
out["realized_false"] = verdict(rec(), realized=False)
out["fp_mismatch"] = verdict(rec(provenance_param_fingerprint="deadbeef"))
out["fp_absent"] = verdict(drop("provenance_param_fingerprint"))
out["no_load"] = verdict(rec(), load={})
out["padded"] = verdict(rec(n_padded=3))
out["padded_absent"] = verdict(drop("n_padded"))
# --- MAJOR 2: fixture identity is CONTENT identity, and the 24GB DB is never read -----------
out["fx_self"] = ev._is_fixture_db(ev.FIXTURE_DB)
out["fx_copy"] = ev._is_fixture_db(cfg["copy"])
out["fx_truncated"] = ev._is_fixture_db(cfg["truncated"])
out["fx_missing"] = ev._is_fixture_db(os.path.join(cfg["work"], "does_not_exist.db"))
out["fx_dir"] = ev._is_fixture_db(cfg["work"])
out["fx_hugefile"] = out["fx_hugefile_seconds"] = out["fx_hugefile_size"] = None
if cfg["hugefile"]:
    t0 = time.time()
    out["fx_hugefile"] = ev._is_fixture_db(cfg["hugefile"])
    out["fx_hugefile_seconds"] = time.time() - t0
    out["fx_hugefile_size"] = os.path.getsize(cfg["hugefile"])
# --- MINOR: a historical pre-886 universe_pin must not render a null into the pasted row ----
with open(cfg["doc"]) as fh:
    doc = json.load(fh)
doc["universe_pin"].pop("verified", None)
doc["universe_pin"].pop("verified_reason", None)
out["historical_row"] = ev.make_ledger_row(doc)
print(json.dumps(out))
'''


def _write_frame(path, symbols):
    with open(path, "w") as fh:
        json.dump({"frame_symbols": list(symbols), "n_assets": len(symbols),
                   "source": "test_universe_pin_886.py", "train_window": [START, END],
                   "db": FIXTURE_DB}, fh)


def _write_periods_file(path, symbols):
    """train_jax_fast.py:227 / eval_vector.py --periods-file format: periods[0] carries
    start_date/end_date/assets. TASK_886 scope 3's NEW "periods-file" mechanism value."""
    with open(path, "w") as fh:
        json.dump({"periods": [{"start_date": START, "end_date": END,
                                 "assets": list(symbols)}]}, fh)


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


def _leaf_dir(ckpt_run_dir):
    """The one orbax leaf `imprint.py`'s trainer.save_checkpoint wrote inside a run dir —
    `<run_dir>/checkpoint_<step>` (flax's own naming)."""
    leaves = [d for d in os.listdir(ckpt_run_dir) if d.startswith("checkpoint_")]
    assert len(leaves) == 1, f"expected exactly one checkpoint_* leaf in {ckpt_run_dir}: {leaves}"
    return os.path.join(ckpt_run_dir, leaves[0])


def main():
    failures = []
    before = (os.path.getsize(FIXTURE_DB), os.path.getmtime(FIXTURE_DB))
    work = tempfile.mkdtemp(prefix="t886_universe_pin_cert_")
    try:
        frame_a = os.path.join(work, "frame_A.json")
        _write_frame(frame_a, FRAME_SYMBOLS)

        print("[886] training checkpoint PROBE under FRAME_A (imprint.py emits its OWN "
              "provenance beside it, TASK_874, WITH param_fingerprint since TASK_888/889) ...")
        ckpt = os.path.join(work, "ckpt_probe")
        r = _run([IMPRINT_PY, "--db", FIXTURE_DB, "--start", START, "--end", END,
                  "--bar-minutes", str(BAR_MINUTES), "--n-assets", str(N_ASSETS),
                  "--panel-frame-from-checkpoint", frame_a, "--panel-frame-source", "minute",
                  "--imprint-mapping", "weight-native", "--iters", str(TRAIN_ITERS),
                  "--batch-size", "32", "--seed", "0", "--vector-weight-rebalance",
                  "--ckpt-out", ckpt, "--tag", "t886", "--out",
                  os.path.join(work, "imprint_probe.json")])
        if r.returncode != 0:
            raise RuntimeError(f"imprint.py training (probe) failed (exit {r.returncode}):\n"
                               f"{r.stderr[-3000:]}")
        own_prov = os.path.join(work, "frame_provenance_probe.json")   # imprint.py TASK_874 naming
        if not os.path.exists(own_prov):
            raise RuntimeError(f"expected imprint.py (TASK_874) to emit {own_prov}")
        with open(own_prov, "rb") as fh:
            own_prov_bytes = fh.read()          # restored after every mutation below
        with open(own_prov) as fh:
            doc_prov = json.load(fh)
        if "param_fingerprint" not in doc_prov:
            raise RuntimeError(
                f"setup assumption broken: {own_prov} carries no param_fingerprint — "
                f"imprint.py (TASK_888/889) is expected to write one unconditionally at save "
                f"time. doc={doc_prov}")
        fp1 = doc_prov["param_fingerprint"]
        leaf = _leaf_dir(ckpt)
        print(f"    run dir: {ckpt}\n    orbax leaf: {leaf}\n    fp1={fp1[:16]}...")

        def _restore_prov():
            with open(own_prov, "wb") as fh:
                fh.write(own_prov_bytes)

        print("[886] training checkpoint PROBE2 (different --seed => different weights, same "
              "architecture) for the fingerprint-inequality leg (3b-) ...")
        ckpt2 = os.path.join(work, "ckpt_probe2")
        r = _run([IMPRINT_PY, "--db", FIXTURE_DB, "--start", START, "--end", END,
                  "--bar-minutes", str(BAR_MINUTES), "--n-assets", str(N_ASSETS),
                  "--panel-frame-from-checkpoint", frame_a, "--panel-frame-source", "minute",
                  "--imprint-mapping", "weight-native", "--iters", str(TRAIN_ITERS),
                  "--batch-size", "32", "--seed", "1", "--vector-weight-rebalance",
                  "--ckpt-out", ckpt2, "--tag", "t886b", "--out",
                  os.path.join(work, "imprint_probe2.json")])
        if r.returncode != 0:
            raise RuntimeError(f"imprint.py training (probe2) failed (exit {r.returncode}):\n"
                               f"{r.stderr[-3000:]}")
        with open(os.path.join(work, "frame_provenance_probe2.json")) as fh:
            fp3 = json.load(fh)["param_fingerprint"]
        if fp3 == fp1:
            raise RuntimeError("setup assumption broken: probe and probe2 fingerprint IDENTICAL "
                               "(different --seed should give different weights)")

        # ---- gate (0c): fixture caveat keyed on the DB opened, not --tag --------------------
        print("[886] gate (0c) — the FIXTURE-SMOKE ledger_row wording must survive a custom "
              "--tag that does NOT start with 'fixture' ...")
        custom_tag = "customtag_not_fixture_prefixed"
        r = _run([EVAL_PY] + BASE + ["--checkpoint", ckpt,
                                     "--panel-frame-from-checkpoint", own_prov,
                                     "--panel-frame-source", "minute",
                                     "--tag", custom_tag, "--out-dir", work])
        doc = _one_doc(work, custom_tag)
        ok = (r.returncode == 0 and doc is not None
              and doc.get("db") == FIXTURE_DB
              and "FIXTURE-SMOKE" in doc.get("ledger_row", "")
              and "Scheduled-run numbers" not in doc.get("ledger_row", "")
              and not custom_tag.startswith("fixture"))   # sanity: the mutation is REAL
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (0c) fixture caveat survives a "
              f"non-'fixture'-prefixed --tag on fixture data")
        if not ok:
            failures.append(
                f"gate (0c): expected ledger_row to carry FIXTURE-SMOKE wording (keyed on "
                f"db={FIXTURE_DB!r}, not --tag={custom_tag!r}). doc.ledger_row="
                f"{(doc or {}).get('ledger_row')}\nstderr tail:\n{r.stderr[-1500:]}")

        print("[886] gate (0c-2) — ... and must survive a SYMLINK spelling of the fixture "
              "path (realpath, not abspath) ...")
        db_link = os.path.join(work, "fixture_link.db")
        os.symlink(FIXTURE_DB, db_link)
        r = _run([EVAL_PY] + [a if a != FIXTURE_DB else db_link for a in BASE]
                 + ["--checkpoint", ckpt, "--allow-unpinned-universe",
                    "--tag", "g0c2", "--out-dir", work])
        doc = _one_doc(work, "g0c2")
        ok = (r.returncode == 0 and doc is not None
              and "FIXTURE-SMOKE" in doc.get("ledger_row", "")
              and "Scheduled-run numbers" not in doc.get("ledger_row", ""))
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (0c-2) fixture caveat under a symlinked "
              f"--db")
        if not ok:
            failures.append(
                f"gate (0c-2): expected FIXTURE-SMOKE wording for --db {db_link!r} (a symlink "
                f"to the pinned fixture). returncode={r.returncode} ledger_row="
                f"{(doc or {}).get('ledger_row')}")

        print("[886] gate (0c-3) — ... and must survive a byte-identical COPY of the fixture at "
              "an unrelated path (CONTENT identity, not path identity) ...")
        # 886 fix pass, MAJOR 2. `realpath` alone reads False for a copy, so this run used to
        # ship "Scheduled-run numbers; PASS verdict applied by the planner (gate ladder)" for
        # what is untrained fixture near-noise — strictly WORSE than HEAD's --tag-keyed check
        # for exactly this input. This repo duplicates the gitignored fixture as a matter of
        # routine (the HEAD-control worktree A/B rig), so a copy is not a contrived input.
        db_copy = os.path.join(work, "fixture_copy.db")
        shutil.copyfile(FIXTURE_DB, db_copy)
        r = _run([EVAL_PY] + [a if a != FIXTURE_DB else db_copy for a in BASE]
                 + ["--checkpoint", ckpt, "--allow-unpinned-universe",
                    "--tag", "g0c3", "--out-dir", work])
        doc = _one_doc(work, "g0c3")
        ok = (r.returncode == 0 and doc is not None
              and doc.get("db") == db_copy          # the copy really was the DB opened
              and "FIXTURE-SMOKE" in doc.get("ledger_row", "")
              and "Scheduled-run numbers" not in doc.get("ledger_row", ""))
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (0c-3) fixture caveat under a byte-"
              f"identical COPY of the fixture")
        if not ok:
            failures.append(
                f"gate (0c-3): expected FIXTURE-SMOKE wording for --db {db_copy!r} (a "
                f"byte-identical copy of the pinned fixture). returncode={r.returncode} "
                f"db={(doc or {}).get('db')!r} ledger_row={(doc or {}).get('ledger_row')}\n"
                f"stderr tail:\n{r.stderr[-1500:]}")

        # ---- gate (i): every spelling agrees; a reversed pin refuses under all -------------
        print("[886] gate (i) — run-dir, orbax-leaf and SYMLINK --checkpoint spellings of the "
              "SAME checkpoint must produce IDENTICAL universe_pin output (Hole 1: "
              "_own_provenance_path_for resolves the RUN DIR first) ...")
        ckpt_link = os.path.join(work, "link_to_probe")
        os.symlink(ckpt, ckpt_link)
        r_run = _run([EVAL_PY] + BASE + ["--checkpoint", ckpt,
                                         "--panel-frame-from-checkpoint", own_prov,
                                         "--panel-frame-source", "minute",
                                         "--tag", "g_i_run", "--out-dir", work])
        r_leaf = _run([EVAL_PY] + BASE + ["--checkpoint", leaf,
                                          "--panel-frame-from-checkpoint", own_prov,
                                          "--panel-frame-source", "minute",
                                          "--tag", "g_i_leaf", "--out-dir", work])
        r_link = _run([EVAL_PY] + BASE + ["--checkpoint", ckpt_link,
                                          "--panel-frame-from-checkpoint", own_prov,
                                          "--panel-frame-source", "minute",
                                          "--tag", "g_i_link", "--out-dir", work])
        doc_run, doc_leaf, doc_link = (_one_doc(work, t) for t in
                                       ("g_i_run", "g_i_leaf", "g_i_link"))
        up_run = (doc_run or {}).get("universe_pin")
        up_leaf = (doc_leaf or {}).get("universe_pin")
        up_link = (doc_link or {}).get("universe_pin")
        ok = (r_run.returncode == 0 and r_leaf.returncode == 0 and r_link.returncode == 0
              and up_run is not None and up_run == up_leaf == up_link
              and up_run.get("checkpoint_provenance_found") is True
              and up_run.get("mechanism") == "checkpoint-provenance"
              and up_run.get("realized_matches_requested") is True
              and up_run.get("n_padded") == 0
              # This checkpoint's own provenance carries a fingerprint that MATCHES what was
              # actually loaded (same freshly-trained checkpoint) — so this run CERTIFIES.
              # Gate (3b+) below proves both directions explicitly with a controlled mutation;
              # this leg proves it also holds for the ORDINARY, unmutated path.
              and up_run.get("verified") is True
              and up_run.get("certified") is True
              and (doc_run or {}).get("universe_pinned") is True
              and up_run.get("verified_reason") == "certified"
              and up_run.get("provenance_param_fingerprint") == fp1
              and (doc_run or {}).get("checkpoint_load", {}).get("param_fingerprint") == fp1)
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (i) same universe_pin under all three "
              f"spellings; CERTIFIED")
        if not ok:
            failures.append(
                f"gate (i): expected identical universe_pin under run-dir/leaf/symlink "
                f"spellings, verified=true reason='certified'. up_run={up_run}\n"
                f"up_leaf={up_leaf}\nup_link={up_link}\n"
                f"run stderr:\n{r_run.stderr[-1200:]}\nleaf stderr:\n{r_leaf.stderr[-1200:]}\n"
                f"link stderr:\n{r_link.stderr[-1200:]}")

        frame_reversed = os.path.join(work, "frame_reversed.json")
        _write_frame(frame_reversed, [FRAME_SYMBOLS[1], FRAME_SYMBOLS[0]] + FRAME_SYMBOLS[2:])
        r_run_rev = _run([EVAL_PY] + BASE + ["--checkpoint", ckpt,
                                             "--panel-frame-from-checkpoint", frame_reversed,
                                             "--panel-frame-source", "minute",
                                             "--tag", "g_i_run_rev", "--out-dir", work])
        r_leaf_rev = _run([EVAL_PY] + BASE + ["--checkpoint", leaf,
                                              "--panel-frame-from-checkpoint", frame_reversed,
                                              "--panel-frame-source", "minute",
                                              "--tag", "g_i_leaf_rev", "--out-dir", work])
        ok = (r_run_rev.returncode != 0 and "MISMATCH" in r_run_rev.stderr
              and "first differing slot: 0" in r_run_rev.stderr
              and not _artifacts(work, "g_i_run_rev")
              and r_leaf_rev.returncode != 0 and "MISMATCH" in r_leaf_rev.stderr
              and "first differing slot: 0" in r_leaf_rev.stderr
              and not _artifacts(work, "g_i_leaf_rev"))
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (i) a REVERSED pin REFUSES under BOTH "
              f"spellings (this is exactly where a pre-886 HEAD exits 0 under the leaf spelling)")
        if not ok:
            failures.append(
                f"gate (i): expected a MISMATCH refusal under BOTH spellings for the reversed "
                f"pin, no artifacts. run: returncode={r_run_rev.returncode} "
                f"stderr={r_run_rev.stderr[-1200:]}\nleaf: returncode={r_leaf_rev.returncode} "
                f"stderr={r_leaf_rev.stderr[-1200:]}")

        # ---- gate (3a): a provenance that NAMES A DIFFERENT CHECKPOINT refuses --------------
        print("[886] gate (3a) — a provenance document that NAMES A DIFFERENT CHECKPOINT must "
              "REFUSE, naming both (the documented non-injective ckpt_foo/foo sibling mapping "
              "reaches this by accident) ...")
        collide = os.path.join(work, "probe")            # -> frame_provenance_probe.json, i.e.
        shutil.copytree(ckpt, collide)                   #    the provenance written for ckpt_probe
        r = _run([EVAL_PY] + BASE + ["--checkpoint", collide,
                                     "--panel-frame-from-checkpoint", own_prov,
                                     "--panel-frame-source", "minute",
                                     "--tag", "g3a", "--out-dir", work])
        ok = (r.returncode != 0
              and "written for a DIFFERENT checkpoint" in r.stderr
              and "ckpt_probe" in r.stderr
              and not _artifacts(work, "g3a"))
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (3a) provenance naming another checkpoint")
        if not ok:
            failures.append(
                f"gate (3a): expected a REFUSAL naming both {collide!r} and the recorded "
                f"checkpoint, and NO artifact. returncode={r.returncode}\n"
                f"stderr tail:\n{r.stderr[-2000:]}")

        # ---- gate (3b): certification is bound to the LOADED PARAMETERS ---------------------
        print("[886] gate (3b-) — a provenance recording a fingerprint that CONTRADICTS the "
              "loaded parameters must REFUSE (fp of a DIFFERENT checkpoint) ...")
        doc_prov_wrong = dict(doc_prov, param_fingerprint=fp3)     # probe2's weights, not probe's
        with open(own_prov, "w") as fh:
            json.dump(doc_prov_wrong, fh)
        try:
            r = _run([EVAL_PY] + BASE + ["--checkpoint", ckpt,
                                         "--panel-frame-from-checkpoint", own_prov,
                                         "--panel-frame-source", "minute",
                                         "--tag", "g3bneg", "--out-dir", work])
            ok = (r.returncode != 0
                  and "DIFFERENT set of parameters" in r.stderr
                  and str(fp3) in r.stderr and str(fp1) in r.stderr
                  and not _artifacts(work, "g3bneg"))
        finally:
            _restore_prov()
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (3b-) contradicting fingerprint REFUSES")
        if not ok:
            failures.append(
                f"gate (3b-): expected a REFUSAL naming both fingerprints and NO artifact. "
                f"returncode={r.returncode}\nstderr tail:\n{r.stderr[-2000:]}")

        print("[886] gate (3b+) — a provenance recording the fingerprint of the parameters "
              "ACTUALLY LOADED certifies: verified/certified/universe_pinned all true and ONE "
              "coherent CERTIFIED statement in ledger_row ...")
        r = _run([EVAL_PY] + BASE + ["--checkpoint", ckpt,
                                     "--panel-frame-from-checkpoint", own_prov,
                                     "--panel-frame-source", "minute",
                                     "--tag", "g3bpos", "--out-dir", work])
        doc = _one_doc(work, "g3bpos")
        up = (doc or {}).get("universe_pin", {})
        row = (doc or {}).get("ledger_row", "")
        ok = (r.returncode == 0 and doc is not None
              and up.get("verified") is True
              and up.get("certified") is True
              and doc.get("universe_pinned") is True
              and up.get("verified_reason") == "certified"
              and up.get("provenance_param_fingerprint") == fp1
              and doc.get("checkpoint_load", {}).get("param_fingerprint") == fp1
              and "UNIVERSE CERTIFIED (TASK_886)" in row
              and "mechanism='checkpoint-provenance'" in row
              and "UNIVERSE NOT CERTIFIED" not in row
              and "UNIVERSE UNVERIFIED" not in row)
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (3b+) matching fingerprint CERTIFIES, "
              f"one coherent statement")
        if not ok:
            failures.append(
                f"gate (3b+): expected exit 0 with verified/certified/universe_pinned all true, "
                f"verified_reason='certified', and exactly ONE coherent CERTIFIED statement in "
                f"ledger_row. returncode={r.returncode} universe_pin={up}\n"
                f"ledger_row={row}\nstderr tail:\n{r.stderr[-2000:]}")

        # ---- gate (echo)/(nul) ---------------------------------------------------------------
        print("[886] gate (echo) — a provenance whose recorded 'checkpoint' is enormous must "
              "still produce a SHORT refusal (content is clipped; counts/paths are not) ...")
        huge = "/nowhere/" + "X" * 500_000
        with open(own_prov, "w") as fh:
            json.dump(dict(doc_prov, checkpoint=huge), fh)
        try:
            r = _run([EVAL_PY] + BASE + ["--checkpoint", ckpt,
                                         "--panel-frame-from-checkpoint", own_prov,
                                         "--panel-frame-source", "minute",
                                         "--tag", "gecho", "--out-dir", work])
            ok = (r.returncode != 0
                  and "written for a DIFFERENT checkpoint" in r.stderr
                  and "clipped" in r.stderr
                  and r.stderr.count("X") < 1_000          # pre-fix: 500001
                  and len(r.stderr) < 20_000               # pre-fix: >500_000
                  and not _artifacts(work, "gecho"))
        finally:
            _restore_prov()
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (echo) refusal echo is bounded "
              f"(stderr {len(r.stderr)} chars)")
        if not ok:
            failures.append(
                f"gate (echo): expected a BOUNDED refusal naming the clip for a 500,000-char "
                f"recorded 'checkpoint'. returncode={r.returncode} "
                f"stderr_len={len(r.stderr)} x_count={r.stderr.count('X')}")

        print("[886] gate (nul) — a provenance 'checkpoint' containing a NUL byte (valid JSON) "
              "must produce a NAMED refusal, not a bare ValueError traceback ...")
        with open(own_prov, "w") as fh:
            json.dump(dict(doc_prov, checkpoint="/nowhere/a\x00b"), fh)
        try:
            r = _run([EVAL_PY] + BASE + ["--checkpoint", ckpt,
                                         "--panel-frame-from-checkpoint", own_prov,
                                         "--panel-frame-source", "minute",
                                         "--tag", "gnul", "--out-dir", work])
            ok = (r.returncode != 0
                  and "written for a DIFFERENT checkpoint" in r.stderr
                  and "not a usable filesystem path" in r.stderr
                  and "ValueError: embedded null byte" not in r.stderr.splitlines()[-1]
                  and not _artifacts(work, "gnul"))
        finally:
            _restore_prov()
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (nul) NUL byte named, not a bare traceback")
        if not ok:
            failures.append(
                f"gate (nul): expected a NAMED refusal for a NUL-containing recorded "
                f"'checkpoint'. returncode={r.returncode}\nstderr tail:\n{r.stderr[-1500:]}")

        # ---- gate (ii) [regression]: DB-absent symbol still REFUSES, never verified:true ----
        print("[886] gate (ii) [regression] — a --symbols list with a DB-absent symbol must "
              "still REFUSE (883R's realized-panel assertion), so verified can never read "
              "true for it ...")
        os.remove(own_prov)   # else the cross-check fires first (same as 883R gate vii)
        try:
            absent = ",".join(FRAME_SYMBOLS[:7] + ["NOPE/USD"])
            r = _run([EVAL_PY] + BASE + ["--checkpoint", ckpt, "--symbols", absent,
                                         "--tag", "g_ii", "--out-dir", work])
            ok = (r.returncode != 0 and "REALIZED panel" in r.stderr
                  and "first differing slot: 7" in r.stderr and "NOPE/USD" in r.stderr
                  and not _artifacts(work, "g_ii"))
        finally:
            _restore_prov()   # imprint.py's original bytes, incl. its "checkpoint" field
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (ii) DB-absent symbol REFUSES")
        if not ok:
            failures.append(
                f"gate (ii): expected a REALIZED-panel REFUSAL naming slot 7 / NOPE/USD, no "
                f"artifact. returncode={r.returncode}\nstderr tail:\n{r.stderr[-2000:]}")

        # ---- gate (iii)/(iv): mechanism present incl. "none"/"periods-file"; named inline ---
        print("[886] gate (iii)/(iv) — universe_pin.mechanism present on EVERY artifact "
              "(incl. \"none\"), and ledger_row names it inline in every case, as ONE "
              "coherent statement ...")
        cases = []   # (tag, extra_argv, expected_mechanism, expect_provenance)
        cases.append(("g_mech_none", ["--allow-unpinned-universe"], "none", False))
        cases.append(("g_mech_symbols", ["--symbols", ",".join(FRAME_SYMBOLS)],
                      "explicit-symbols", False))
        periods_file = os.path.join(work, "periods_886.json")
        _write_periods_file(periods_file, FRAME_SYMBOLS)
        cases.append(("g_mech_periods", ["--periods-file", periods_file],
                      "periods-file", True))
        cases.append(("g_mech_ckptprov",
                      ["--panel-frame-from-checkpoint", own_prov,
                       "--panel-frame-source", "minute"], "checkpoint-provenance", True))
        for tag, extra, expect_mech, expect_prov in cases:
            r = _run([EVAL_PY] + BASE + ["--checkpoint", ckpt, "--tag", tag,
                                         "--out-dir", work] + extra)
            doc = _one_doc(work, tag)
            up = (doc or {}).get("universe_pin", {})
            row = (doc or {}).get("ledger_row", "")
            mech_str = f"mechanism={expect_mech!r}"
            ok = (r.returncode == 0 and doc is not None
                  and up.get("mechanism") == expect_mech
                  and (up.get("provenance_path") is not None) == expect_prov
                  and (up.get("sha256") is not None) == expect_prov
                  and mech_str in row
                  # exactly one universe statement, and it is the honest one for this state
                  and row.count("UNIVERSE ") == 1
                  and ("UNIVERSE CERTIFIED" in row) == bool(up.get("verified")))
            print(f"    [{'PASS' if ok else 'FAIL'}] gate (iii)/(iv) mechanism={expect_mech!r} "
                  f"verified={up.get('verified')}")
            if not ok:
                failures.append(
                    f"gate (iii)/(iv) [{tag}]: expected mechanism={expect_mech!r}, "
                    f"provenance_path/sha256 present={expect_prov}, {mech_str!r} inline in "
                    f"ledger_row, and EXACTLY ONE 'UNIVERSE ...' statement agreeing with "
                    f"universe_pin.verified. returncode={r.returncode} universe_pin={up}\n"
                    f"ledger_row={row}\nstderr tail:\n{r.stderr[-1500:]}")
        # periods-file and explicit-symbols share 883R's coarse pin_source bucket but must
        # differ on the NEW, finer mechanism field — the whole point of the split (scope 3).
        doc_sym = _one_doc(work, "g_mech_symbols")
        doc_per = _one_doc(work, "g_mech_periods")
        ok = (doc_sym is not None and doc_per is not None
              and doc_sym["universe_pin"]["pin_source"] == doc_per["universe_pin"]["pin_source"]
              == "explicit-symbols"
              and doc_sym["universe_pin"]["mechanism"] != doc_per["universe_pin"]["mechanism"])
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (iii) periods-file vs bare --symbols: "
              f"same legacy pin_source, DIFFERENT (finer) mechanism")
        if not ok:
            failures.append(
                "gate (iii): expected --periods-file and --symbols to share pin_source="
                "'explicit-symbols' (883R, unchanged) but report DIFFERENT `mechanism` values "
                f"(TASK_886 scope 3 split). doc_sym.universe_pin={(doc_sym or {}).get('universe_pin')}"
                f"\ndoc_per.universe_pin={(doc_per or {}).get('universe_pin')}")

        # ---- gate (order-tamper): frame_symbols reversed, param_fingerprint INTACT ----------
        # 886 fix pass, BLOCKER. `param_fingerprint` hashes parameter shape/dtype/bytes per leaf
        # and has ZERO dependence on `frame_symbols` — weights do not encode symbol names. So
        # certification proves WHICH WEIGHTS loaded and nothing about the symbol ORDER they were
        # trained under, and a provenance document whose order is reversed with the fingerprint
        # left intact still certifies while scoring the per-slot weights against DIFFERENT real
        # assets. These legs pin the two honest outcomes: the sentence must not claim the binding
        # it does not have, and where a committed sha256 pin exists the tamper must REFUSE.
        prov_sha = hashlib.sha256(own_prov_bytes).hexdigest()
        pin_file = os.path.join(work, "frame_provenance_probe.sha256")
        pin_file2 = os.path.join(work, "extra_pin.sha256")

        def _tamper_prov():
            """Reverse frame_symbols; leave param_fingerprint (and every other field) alone."""
            d = dict(doc_prov)
            d["frame_symbols"] = list(reversed(doc_prov["frame_symbols"]))
            with open(own_prov, "w") as fh:
                json.dump(d, fh)

        print("[886] gate (order-tamper-a) — frame_symbols REVERSED with param_fingerprint "
              "intact and NO committed sha256 pin: undetectable here, so the CERTIFIED sentence "
              "must not claim the order is bound ...")
        _tamper_prov()
        try:
            r = _run([EVAL_PY] + BASE + ["--checkpoint", ckpt,
                                         "--panel-frame-from-checkpoint", own_prov,
                                         "--panel-frame-source", "minute",
                                         "--tag", "gtampa", "--out-dir", work])
            doc_t = _one_doc(work, "gtampa")
            row_t = (doc_t or {}).get("ledger_row", "")
            doc_clean = _one_doc(work, "g3bpos")
            ok = (r.returncode == 0 and doc_t is not None
                  # it DOES still certify — nothing in this program can see the tamper ...
                  and doc_t["universe_pin"].get("verified") is True
                  and doc_t["universe_pin"].get("checkpoint_provenance_sha256_pin") is None
                  # ... and the number really did move, which is why the sentence matters
                  and doc_t["tiers"]["21"]["sharpe_full"]
                  != doc_clean["tiers"]["21"]["sharpe_full"]
                  # ... so the pasted sentence must name the gap, and must NOT carry the
                  # retired overclaim.
                  and "NOT PROVEN" in row_t
                  and "SELF-ATTESTING" in row_t
                  and "NO committed sha256 pin" in row_t
                  and "slot identity is bound to the checkpoint actually LOADED" not in row_t
                  and row_t.count("UNIVERSE ") == 1)
        finally:
            _restore_prov()
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (order-tamper-a) certified sentence names "
              f"the unproven half instead of claiming it")
        if not ok:
            failures.append(
                f"gate (order-tamper-a): a reversed frame_symbols order with an intact "
                f"param_fingerprint must still complete (it is undetectable without a pin) but "
                f"the CERTIFIED sentence must say the order is SELF-ATTESTING / NOT PROVEN and "
                f"must not claim slot identity is bound. returncode={r.returncode} "
                f"universe_pin={(doc_t or {}).get('universe_pin')}\nledger_row={row_t}\n"
                f"stderr tail:\n{r.stderr[-1500:]}")

        print("[886] gate (order-tamper-b) — the SAME tamper with a committed sha256 pin beside "
              "the document must REFUSE, naming the pin and both digests ...")
        with open(pin_file, "w") as fh:
            fh.write(f"{prov_sha}  {os.path.basename(own_prov)}\n")
        _tamper_prov()
        try:
            r = _run([EVAL_PY] + BASE + ["--checkpoint", ckpt,
                                         "--panel-frame-from-checkpoint", own_prov,
                                         "--panel-frame-source", "minute",
                                         "--tag", "gtampb", "--out-dir", work])
            ok = (r.returncode != 0
                  and "does NOT match its committed sha256 pin" in r.stderr
                  and prov_sha in r.stderr
                  and os.path.basename(pin_file) in r.stderr
                  and not _artifacts(work, "gtampb"))
        finally:
            _restore_prov()
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (order-tamper-b) pinned document tamper "
              f"REFUSES")
        if not ok:
            failures.append(
                f"gate (order-tamper-b): expected a REFUSAL naming the sha256 pin and both "
                f"digests, and NO artifact. returncode={r.returncode}\n"
                f"stderr tail:\n{r.stderr[-2000:]}")

        print("[886] gate (pin-ok) — an UNTAMPERED document matching its committed pin still "
              "certifies, and the row names the pin instead of the no-pin caveat ...")
        # `pin_file` still holds the digest of the RESTORED (original) bytes.
        try:
            # Timed: gate (pin-fifo) below runs this SAME command with a FIFO in the directory
            # and needs a fail-fast timeout budget derived from a measured clean run, not a
            # guessed constant (this suite runs on both the workstation and the server).
            _t0 = time.monotonic()
            r = _run([EVAL_PY] + BASE + ["--checkpoint", ckpt,
                                         "--panel-frame-from-checkpoint", own_prov,
                                         "--panel-frame-source", "minute",
                                         "--tag", "gpinok", "--out-dir", work])
            pinok_secs = time.monotonic() - _t0
            doc_p = _one_doc(work, "gpinok")
            row_p = (doc_p or {}).get("ledger_row", "")
            ok = (r.returncode == 0 and doc_p is not None
                  and doc_p["universe_pin"].get("verified") is True
                  and doc_p["universe_pin"].get("checkpoint_provenance_sha256_pin") is not None
                  and "matches the committed sha256 pin" in row_p
                  and "NO committed sha256 pin" not in row_p
                  and os.path.basename(pin_file) in row_p
                  # the honest limitation is stated in BOTH pin states — the pin bounds the
                  # DOCUMENT, it still does not bind the order to the WEIGHTS.
                  and "NOT PROVEN" in row_p and "SELF-ATTESTING" in row_p
                  and row_p.count("UNIVERSE ") == 1)
        finally:
            pass
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (pin-ok) pinned+untampered certifies and "
              f"names the pin")
        if not ok:
            failures.append(
                f"gate (pin-ok): expected exit 0, verified=true, the pin recorded in "
                f"universe_pin.checkpoint_provenance_sha256_pin and named in ledger_row. "
                f"returncode={r.returncode} universe_pin={(doc_p or {}).get('universe_pin')}\n"
                f"ledger_row={row_p}\nstderr tail:\n{r.stderr[-1500:]}")

        print("[886] gate (pin-ambig) — two committed pins disagreeing about the SAME document "
              "must REFUSE rather than pick one ...")
        with open(pin_file2, "w") as fh:
            fh.write(f"{'0' * 64}  {os.path.basename(own_prov)}\n")
        try:
            r = _run([EVAL_PY] + BASE + ["--checkpoint", ckpt,
                                         "--panel-frame-from-checkpoint", own_prov,
                                         "--panel-frame-source", "minute",
                                         "--tag", "gpinamb", "--out-dir", work])
            ok = (r.returncode != 0
                  and "two committed sha256 pins disagree" in r.stderr
                  and os.path.basename(pin_file) in r.stderr
                  and os.path.basename(pin_file2) in r.stderr
                  and not _artifacts(work, "gpinamb"))
        finally:
            os.remove(pin_file2)
            os.remove(pin_file)          # later gates must see the unpinned state again
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (pin-ambig) disagreeing pins REFUSE")
        if not ok:
            failures.append(
                f"gate (pin-ambig): expected a REFUSAL naming both pin files and NO artifact. "
                f"returncode={r.returncode}\nstderr tail:\n{r.stderr[-2000:]}")

        # ---- gate (pin-sync): the SYNCHRONIZED tamper — document AND pin both updated -------
        # 886 fix pass, cycle-3 BLOCKER 2. (order-tamper-b) above only exercises the ASYMMETRIC
        # tamper (document moved, pin left stale), which is why the certified sentence was able
        # to claim for a whole cycle that "tampering with that order is ruled out for THIS
        # document". It is not: the .sha256 is a sibling file behind the SAME write permission
        # as the JSON — recomputing it costs the tamperer one extra line. This leg pins the
        # HONEST post-fix behaviour: the run DOES certify (nothing here can see it), the number
        # DOES move, and the pasted sentence must therefore describe what the pin actually
        # catches (an asymmetric change) and name the gap instead of claiming it closed.
        print("[886] gate (pin-sync) — frame_symbols REVERSED *and* the committed sha256 pin "
              "recomputed to match: undetectable, so the CERTIFIED sentence must not claim "
              "tampering is ruled out ...")
        _tamper_prov()
        with open(own_prov, "rb") as fh:
            tampered_sha = hashlib.sha256(fh.read()).hexdigest()
        with open(pin_file, "w") as fh:                  # the tamperer updates BOTH
            fh.write(f"{tampered_sha}  {os.path.basename(own_prov)}\n")
        try:
            r = _run([EVAL_PY] + BASE + ["--checkpoint", ckpt,
                                         "--panel-frame-from-checkpoint", own_prov,
                                         "--panel-frame-source", "minute",
                                         "--tag", "gpinsync", "--out-dir", work])
            doc_s = _one_doc(work, "gpinsync")
            row_s = (doc_s or {}).get("ledger_row", "")
            doc_clean = _one_doc(work, "g3bpos")
            ok = (r.returncode == 0 and doc_s is not None
                  # the pin was found, matched, and recorded — and it caught NOTHING ...
                  and doc_s["universe_pin"].get("verified") is True
                  and doc_s["universe_pin"].get("checkpoint_provenance_sha256_pin") is not None
                  # ... while the number really did move, exactly as in (order-tamper-a)
                  and doc_s["tiers"]["21"]["sharpe_full"]
                  != doc_clean["tiers"]["21"]["sharpe_full"]
                  # ... so the pasted sentence must state the check's ACTUAL reach, and must
                  # never carry the retired "ruled out" overclaim.
                  and "ruled out" not in row_s
                  and "changed WITHOUT its pin changing" in row_s
                  and "does NOT defend against a writer who updates BOTH" in row_s
                  and "NOT PROVEN" in row_s and "SELF-ATTESTING" in row_s
                  and row_s.count("UNIVERSE ") == 1)
        finally:
            _restore_prov()
            os.remove(pin_file)
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (pin-sync) synchronized tamper certifies "
              f"but the sentence describes what the pin ACTUALLY catches")
        if not ok:
            failures.append(
                f"gate (pin-sync): a reversed frame_symbols order WITH the sha256 pin "
                f"recomputed to match must still complete (nothing here can detect it) but the "
                f"CERTIFIED sentence must say the pin only catches a document that changed "
                f"WITHOUT its pin changing, must say it does NOT defend against a writer who "
                f"updates BOTH, and must not claim tampering is 'ruled out'. "
                f"returncode={r.returncode} universe_pin={(doc_s or {}).get('universe_pin')}\n"
                f"ledger_row={row_s}\nstderr tail:\n{r.stderr[-1500:]}")

        # ---- gate (pin-fifo): a NON-REGULAR *.sha256 must not be opened ---------------------
        # 886 fix pass, cycle-3 BLOCKER 1 — D38's cycle-2 hang-guard defect class, recreated in
        # the pin lookup. `_provenance_sha256_pin` scans the document's directory and opens
        # EVERY *.sha256 in it before it can know whether any of them names the document, so a
        # FIFO under ANY name there hung every certification of every document in that directory
        # (reproduced twice, `timeout 8` -> exit 124: once as <doc-stem>.sha256, once as an
        # unrelated name). Both spellings are placed at once — the guard is name-blind, so this
        # is the strictly stronger assertion. Bounded by its OWN timeout, derived from the
        # measured wall time of the identical clean run above, so a regression FAILS FAST
        # instead of wedging the suite for the driver's full 280s default.
        print("[886] gate (pin-fifo) — a FIFO named *.sha256 beside the document must be "
              "SKIPPED unopened (a hang is not a verdict), under both an unrelated name and the "
              "<doc-stem> name ...")
        fifos = [os.path.join(work, "unrelated_junk.sha256"),
                 os.path.join(work, "frame_provenance_probe.sha256")]
        for p in fifos:
            os.mkfifo(p)
        fifo_timeout = max(60.0, 4.0 * pinok_secs)
        timed_out = False
        r = None
        try:
            r = _run([EVAL_PY] + BASE + ["--checkpoint", ckpt,
                                         "--panel-frame-from-checkpoint", own_prov,
                                         "--panel-frame-source", "minute",
                                         "--tag", "gfifo", "--out-dir", work],
                     timeout=fifo_timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            for p in fifos:
                os.remove(p)
        doc_f = _one_doc(work, "gfifo") if not timed_out else None
        row_f = (doc_f or {}).get("ledger_row", "")
        ok = (not timed_out and r is not None and r.returncode == 0 and doc_f is not None
              # skipped, not believed: a pin file that cannot be read names nothing, so the
              # document is honestly UNPINNED rather than hung, guessed, or silently passed.
              and doc_f["universe_pin"].get("verified") is True
              and doc_f["universe_pin"].get("checkpoint_provenance_sha256_pin") is None
              and "NO committed sha256 pin" in row_f)
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (pin-fifo) non-regular pin candidates are "
              f"skipped unopened (no hang; timeout budget {fifo_timeout:.0f}s, "
              f"timed_out={timed_out})")
        if not ok:
            failures.append(
                f"gate (pin-fifo): a FIFO named *.sha256 beside the provenance document must be "
                f"skipped unopened — the run must finish within {fifo_timeout:.0f}s and record "
                f"the honest no-pin state. timed_out={timed_out} "
                f"returncode={getattr(r, 'returncode', None)} "
                f"universe_pin={(doc_f or {}).get('universe_pin')}\nledger_row={row_f}\n"
                f"stderr tail:\n{getattr(r, 'stderr', '')[-1500:]}")

        # ---- DIRECT-PREDICATE gates (886 fix pass) ------------------------------------------
        print("[886] running the DIRECT-PREDICATE probe (_certification_verdict / "
              "_is_fixture_db / make_ledger_row called with constructed inputs) ...")
        probe_py = os.path.join(work, "probe_886.py")
        with open(probe_py, "w") as fh:
            fh.write(_PREDICATE_PROBE)
        truncated = os.path.join(work, "fixture_truncated.db")
        shutil.copyfile(FIXTURE_DB, truncated)
        with open(truncated, "r+b") as fh:                 # same bytes, ONE byte shorter
            fh.truncate(os.path.getsize(FIXTURE_DB) - 1)
        # A 24 GiB SPARSE file standing in for `trading_data.db`: proves the size gate decides
        # before a byte is read (the protected 24GB source of record must never be hashed or
        # streamed by a LEDGER-caveat check). Skipped, not failed, if the fs refuses the hole.
        hugefile = os.path.join(work, "huge_sparse.db")
        try:
            with open(hugefile, "wb") as fh:
                fh.truncate(24 * 1024 ** 3)
        except OSError:
            hugefile = None
        arts = _artifacts(work, "g_mech_none")
        cfg_path = os.path.join(work, "probe_cfg.json")
        with open(cfg_path, "w") as fh:
            json.dump({"work": work, "copy": db_copy, "truncated": truncated,
                       "hugefile": hugefile,
                       "doc": os.path.join(work, arts[0])}, fh)
        pr = _run([probe_py, EVAL_PY, cfg_path])
        probe = None
        if pr.returncode == 0 and pr.stdout.strip():
            probe = json.loads(pr.stdout.strip().splitlines()[-1])
        if probe is None:
            failures.append(
                f"DIRECT-PREDICATE probe failed to run (exit {pr.returncode}).\n"
                f"stdout tail:\n{pr.stdout[-1500:]}\nstderr tail:\n{pr.stderr[-2500:]}")
        else:
            # The control MUST certify — otherwise every negative below is vacuous.
            ctl_ok = probe["control"] == [True, "certified"]
            print(f"    [{'PASS' if ctl_ok else 'FAIL'}] (control) a record with all conjuncts "
                  f"true CERTIFIES: {probe['control']}")
            if not ctl_ok:
                failures.append(
                    f"DIRECT-PREDICATE control: a record with every conjunct forced true must "
                    f"return (True, 'certified') — otherwise the negative legs below prove "
                    f"nothing. got {probe['control']}")

            print("[886] gate (mech) — `mechanism` must be a LIVE conjunct: an UNPINNED record "
                  "with the other four conjuncts forced TRUE must NOT certify ...")
            ok = (ctl_ok
                  and probe["mech_none"][0] is False
                  and "unpinned" in probe["mech_none"][1]
                  and probe["mech_absent"][0] is False
                  and probe["mech_null"][0] is False
                  and probe["mech_unknown"][0] is False
                  # ... and the two OTHER certifiable mechanisms still do certify (the conjunct
                  # is an allowlist, not a blanket refusal).
                  and probe["mech_periods"] == [True, "certified"]
                  and probe["mech_explicit"] == [True, "certified"])
            print(f"    [{'PASS' if ok else 'FAIL'}] gate (mech) mechanism='none'/absent/null/"
                  f"unknown all REFUSE; periods-file + explicit-symbols still certify")
            if not ok:
                failures.append(
                    f"gate (mech): `mechanism` is not a live conjunct of _certification_verdict "
                    f"(a hand-built unpinned/mechanism-less record certified). "
                    f"none={probe['mech_none']} absent={probe['mech_absent']} "
                    f"null={probe['mech_null']} unknown={probe['mech_unknown']} "
                    f"periods={probe['mech_periods']} explicit={probe['mech_explicit']}")

            print("[886] gate (order) — the AGREEMENT half of conjunct (1) must be live too: "
                  "`order_exact` false/null/absent must NOT certify ...")
            ok = (ctl_ok
                  and probe["order_false"][0] is False
                  and probe["order_null"][0] is False
                  and probe["order_absent"][0] is False)
            print(f"    [{'PASS' if ok else 'FAIL'}] gate (order) order_exact false/null/absent "
                  f"all REFUSE")
            if not ok:
                failures.append(
                    f"gate (order): `order_exact` is not a live conjunct — a record recording a "
                    f"provenance that was FOUND but did NOT agree order-exact certified. "
                    f"false={probe['order_false']} null={probe['order_null']} "
                    f"absent={probe['order_absent']}")

            print("[886] gate (conj) — the three already-live conjuncts still refuse when the "
                  "probe drives them directly (this probe can see failures at all) ...")
            ok = (ctl_ok
                  and probe["prov_false"][0] is False
                  and probe["realized_false"][0] is False
                  and probe["fp_mismatch"][0] is False
                  and probe["fp_absent"][0] is False
                  and probe["no_load"][0] is False
                  and probe["padded"][0] is False
                  and probe["padded_absent"][0] is False)
            print(f"    [{'PASS' if ok else 'FAIL'}] gate (conj) provenance / realized-panel / "
                  f"fingerprint / n_padded conjuncts all live")
            if not ok:
                failures.append(
                    f"gate (conj): a pre-existing conjunct failed to refuse. "
                    f"prov={probe['prov_false']} realized={probe['realized_false']} "
                    f"fp_mismatch={probe['fp_mismatch']} fp_absent={probe['fp_absent']} "
                    f"no_load={probe['no_load']} padded={probe['padded']} "
                    f"padded_absent={probe['padded_absent']}")

            print("[886] gate (0c-4) — _is_fixture_db is CONTENT identity, and a 24GiB file is "
                  "decided by SIZE before a byte is read (trading_data.db is never hashed) ...")
            huge_ok = (hugefile is None or
                       (probe["fx_hugefile"] is False and probe["fx_hugefile_seconds"] < 1.0))
            ok = (probe["fx_self"] is True
                  and probe["fx_copy"] is True
                  and probe["fx_truncated"] is False       # same bytes, one shorter => not it
                  and probe["fx_missing"] is False
                  and probe["fx_dir"] is False
                  and huge_ok)
            print(f"    [{'PASS' if ok else 'FAIL'}] gate (0c-4) self/copy True; truncated/"
                  f"missing/dir False; 24GiB file {probe['fx_hugefile']} in "
                  f"{probe['fx_hugefile_seconds']}s (size {probe['fx_hugefile_size']})")
            if not ok:
                failures.append(
                    f"gate (0c-4): _is_fixture_db must be True for the fixture and for a "
                    f"byte-identical COPY, False for a truncated copy / missing path / "
                    f"directory, and must reject a 24GiB file on SIZE in well under a second. "
                    f"self={probe['fx_self']} copy={probe['fx_copy']} "
                    f"truncated={probe['fx_truncated']} missing={probe['fx_missing']} "
                    f"dir={probe['fx_dir']} huge={probe['fx_hugefile']} "
                    f"huge_seconds={probe['fx_hugefile_seconds']}")

            print("[886] gate (hist) [MINOR] — a HISTORICAL pre-886 universe_pin record (no "
                  "`verified`/`verified_reason` keys) must not render a null into the pasted "
                  "LEDGER row ...")
            hist_row = probe["historical_row"]
            ok = ("UNIVERSE NOT CERTIFIED (TASK_886)" in hist_row
                  and "NOT certified — None" not in hist_row
                  and "pre-TASK_886 record" in hist_row)
            print(f"    [{'PASS' if ok else 'FAIL'}] gate (hist) pre-886 record names its state "
                  f"instead of printing None")
            if not ok:
                failures.append(
                    f"gate (hist): make_ledger_row fed a pre-886 universe_pin (no `verified`/"
                    f"`verified_reason`) must name that state, never emit 'NOT certified — "
                    f"None'. row={hist_row}")

    finally:
        shutil.rmtree(work, ignore_errors=True)

    after = (os.path.getsize(FIXTURE_DB), os.path.getmtime(FIXTURE_DB))
    if before != after:
        failures.append(f"fixtures/trading_fixture.db changed size/mtime during this run "
                        f"({before} -> {after}) — INVARIANT VIOLATION, never regenerate it.")
    else:
        print(f"[886] fixture unchanged: size={before[0]} mtime={before[1]}")

    print("=" * 84)
    if failures:
        print(f"[886] FAIL — {len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[886] PASS — gates (0c), (0c-2), (0c-3), (0c-4), (i), (3a), (3b-), (3b+), (echo), "
          "(nul), (ii), (iii)/(iv), (order-tamper-a), (order-tamper-b), (pin-ok), (pin-ambig), "
          "(pin-sync), (pin-fifo), (mech), (order), (conj) and (hist) green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
