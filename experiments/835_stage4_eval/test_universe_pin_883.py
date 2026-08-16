#!/usr/bin/env python3
"""TASK_883R — regression test for eval_vector.py's universe-identity fail-loud gates.

Context (TASK_882 -> TASK_883 -> run-29 HALT -> D36 re-cut as 883R): a missing
`--panel-frame-from-checkpoint` silently routes `eval_vector.py` through TODAY's
coverage-ranked universe instead of the checkpoint's training universe. The encoder is
positionally universe-locked (slot k IS asset k), so this scrambles slot->asset identity and
still prints a paste-ready LEDGER row, with no warning anywhere in the output (measured:
63/64 positions wrong, 24 symbols absent, first divergence at decision 0). `imprint.py`
already refuses this class of run; TASK_883 closed the SAME hole in `eval_vector.py`'s
checkpoint-grading entry path — and run 29 then found two fail-open holes IN the fix
(argv-spelling bypass; explicit-symbols stamping a true label from the command line rather
than the realized panel). D36 re-cut the row by CERTIFICATION CAPACITY: 883R ships only what
can REFUSE or DISCLAIM and nothing that can BLESS. This file pins the acceptance gates as
re-cut:

  (i)   an unpinned `--checkpoint` grading run REFUSES before any panel is built, naming the
        checkpoint, the missing flag and the escape hatch.
  (ii)  a pin that disagrees with the checkpoint's OWN provenance (TASK_874's
        `frame_provenance_<name>.json` sibling, emitted automatically by `imprint.py`) REFUSES,
        naming the first differing slot and both symbols.
  (iii) `--allow-unpinned-universe` runs to completion, prints the warning, and the emitted
        JSON carries `"universe_pinned": false`.
  (v)   THE PROVENANCE STATE TABLE (fix pass). `_read_provenance_frame` used to return `None`
        for BOTH "genuinely absent" and "present but unreadable/malformed", which silently
        downgraded gate (ii) to a no-op while the artifact looked pinned — this task's own
        thesis (absence read as permission) reproduced inside the guard written to close it.
        The table below pins every state, so only a GENUINELY ABSENT file may proceed.
  (vi)  THE EXPLICIT-SYMBOLS PIN SOURCE (fix pass). `build_panel` has a SECOND, equally
        order-preserving universe mechanism — `specific_symbols=args.symbols`, populated by
        `--symbols` or `--periods-file`'s "assets" list — and it is the one every
        `experiments/837_walkforward/walkforward.py` eval job uses, against train_jax_fast.py
        checkpoints that never emit a frame-provenance sibling. An explicitly-named universe
        is a KNOWN request: it must run, must record `pin_source: "explicit-symbols"` and
        whether anything independent verified it, and must still REFUSE when it order-exactly
        disagrees with a provenance the checkpoint does carry.
  (vii) THE REALIZED-PANEL EQUALITY ASSERTION (883R NEW, D36 — the hole-2 mechanism). After
        `build_panel` returns, the run REFUSES unless the realized slot->symbol list equals
        the requested universe order-exact. Pinned here on the two run-29 hole-2 modes: a
        DB-absent symbol in the explicit list (silently dropped, every later slot shifted —
        run 29 measured 6/8 slots wrong and a paste-ready row at net@21 −3.953) and
        `n_symbols < --n-assets` (duplicate #pad slots). Both now REFUSE, naming the first
        differing slot and both symbols.
  (viii) CYCLE-2 FIX-PASS legs. (a)/(b) a DUPLICATE explicit symbol refuses on DISTINCTNESS
        — before the cross-check, with and without provenance — naming the symbol and the
        slots it occupies (a duplicated request realizes faithfully, so the realized-panel
        assertion cannot see the slot collapse; the identical frame via a pin FILE was
        already refused by `_load_frame_provenance`). (c) a FIFO at the auto-derived
        provenance path refuses FAST (stat-before-open; the pre-fix reader hung forever).
        (d) refusal text echoes a key COUNT and slot INDICES only — never key names or entry
        values. (e) an empty `--panel-frame-from-checkpoint ""` is reported as
        given-but-empty, not as a missing flag.
  (ix)  CYCLE-3 legs — distinctness judged on the LOADERS' identity function
        (`normalize_symbol_key`, case/slash-insensitive), not exact string. (a)/(b) case-
        and slash-variant duplicates via `--symbols` REFUSE, naming both spellings and the
        slots; (c) a variant-duplicate pin FILE refuses at `_load_frame_provenance`;
        (d) a colluding provenance carrying the same variant frame (exact-string
        cross-check passes) still refuses on distinctness; (e) a valid duplicate-free run
        with mixed-case spellings normalizing to DISTINCT keys still completes,
        disclaimed.

  D36's LOAD-BEARING CONSTRAINT AT THE TIME 883R SHIPPED: `universe_pinned: true` did not
  exist anywhere — every artifact carried `"universe_pinned": false` / `"certified": false`,
  because no emitter wrote a `param_fingerprint` into provenance and the positive claim was
  therefore UNREACHABLE. D36 scoped that freeze as lasting "until the certification row
  lands" (TASK_886). TASK_888/889 made `imprint.py` write `param_fingerprint` at SAVE time
  unconditionally and TASK_886 built the certification predicate that reads it, so a
  checkpoint graded against its OWN matching, freshly-written provenance NOW CERTIFIES (gates
  (iv) and (vi-a) below) — this is the earned positive claim TASK_886 exists to introduce, not
  a regression of 883R's refuse/disclaim discipline (883R's REFUSAL gates — (i)/(ii)/(v)-(ix)
  — are unaffected: certification is additive on top of them, never a relaxation of them).
  Every grading row's `ledger_row` carries the caveat INLINE naming the mechanism, in BOTH the
  certified and not-certified case (D36 Q2: a caveat that does not travel with the pasted text
  is not a caveat; 886 retired 883R's "UNIVERSE UNVERIFIED" sentence for a mechanism-aware
  "UNIVERSE CERTIFIED" / "UNIVERSE NOT CERTIFIED" pair so a certified row never also says "not
  certified" in the same field).

  (iv)  a pin that MATCHES the checkpoint's own provenance is otherwise unaffected: its result
        carries EXACTLY the pre-883 key set plus the two additive keys
        `universe_pinned`/`universe_pin` (TASK_888 additionally added `checkpoint_load` —
        load evidence computed at LOAD time on EVERY run, pinned or not; see EXPECTED_KEYS_PINNED)
        — nothing removed, nothing renamed. (The deep numeric byte-identity claim — this run's
        `tiers`/`corridor` numbers are unchanged from HEAD — was verified separately, once, via
        a `git stash` A/B on this exact checkpoint; seeing it here on every run would require
        keeping a second copy of the pre-883 module around forever, which is not what a
        standing regression test is for.)

Trains ONE real checkpoint on the fixture via `imprint.py` (which, per TASK_874, emits its own
frame provenance next to the checkpoint it writes) and grades it four ways.

CPU / fixture only; no GPU; never touches `trading_data.db`. Run:
    python experiments/835_stage4_eval/test_universe_pin_883.py
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

# The fixture's own 8-symbol minute-slice universe (eval_vector.SMOKE_SYMBOLS / 882's
# FRAME_SYMBOLS) — reused, not re-derived.
FRAME_SYMBOLS = ["XBT/USD", "ETH/USD", "LTC/USD", "XMR/USD",
                 "ZEC/USD", "ETC/USD", "DASH/USD", "REP/USD"]
START, END, BAR_MINUTES, N_ASSETS = "2025-09-14", "2025-12-12", 240, 8
TRAIN_ITERS = 15   # cheap: these checks never depend on the checkpoint being GOOD, only real

# The pre-883 top-level result-dict key set (observed on committed 835/853/856/861/868/869
# artifacts, e.g. runs/882_decisive/results_882decisive_s0_*.json, for a PINNED
# policy-argmax + --vector-weight-rebalance + --teacher-floor run). TASK_883 must add ONLY
# `universe_pinned` / `universe_pin` to this set — nothing removed, nothing renamed.
PRE_883_KEYS = {
    "tag", "checkpoint", "checkpoint_label", "db", "window", "bar_minutes", "window_utc",
    "eval_seed", "panel", "operator", "cost_tiers_bp", "annualization_period_minutes",
    "tiers", "corridor", "teacher_floor", "panel_frame", "net_dependent",
    "out_name", "out_pointer", "ledger_row",
}
# TASK_888 (D38 scope 3): `checkpoint_load` — resolved_path/step/param_fingerprint computed at
# LOAD time, always (independent of provenance/pinning; every result carries it, pinned or
# not). Additive, same as 883's own two keys were.
EXPECTED_KEYS_PINNED = PRE_883_KEYS | {"universe_pinned", "universe_pin", "checkpoint_load"}


def _write_frame(path, symbols):
    with open(path, "w") as fh:
        json.dump({"frame_symbols": list(symbols), "n_assets": len(symbols),
                   "source": "test_universe_pin_883.py", "train_window": [START, END],
                   "db": FIXTURE_DB}, fh)


def _run(argv, timeout=280):
    return subprocess.run([sys.executable] + argv, cwd=_REPO, capture_output=True, text=True,
                          timeout=timeout)


def _artifacts(work, tag):
    return [f for f in os.listdir(work) if f.startswith(f"results_{tag}_") and f.endswith(".json")]


# ---- gate (v): the checkpoint-own-provenance STATE TABLE ---------------------------------
# Each entry mutates the checkpoint's OWN `frame_provenance_probe.json` into one state, then
# grades with a pin that MATCHES the checkpoint's real frame. Expectation is one of:
#   "proceed" — exit 0, artifact emitted, `checkpoint_provenance_found: false` recorded
#               honestly (ONLY the genuinely-absent file may do this)
#   "unusable" — REFUSE by name: present but unreadable/unparseable/schema-invalid
#   "mismatch" — readable and well-formed, so it reaches the order-exact comparison and
#                refuses THERE (duplicates cannot agree with a duplicate-free pin)
def _st_absent(p, intact):
    os.remove(p)


def _st_empty(p, intact):
    open(p, "w").close()


def _st_truncated(p, intact):
    with open(p, "wb") as fh:
        fh.write(intact[:40])


def _st_list(p, intact):
    with open(p, "w") as fh:
        fh.write("[1, 2, 3]")


def _st_null(p, intact):
    with open(p, "w") as fh:
        fh.write("null")


def _st_no_key(p, intact):
    with open(p, "w") as fh:
        json.dump({"n_assets": N_ASSETS, "db": FIXTURE_DB, "source": "state-table"}, fh)


def _st_empty_frame(p, intact):
    with open(p, "w") as fh:
        json.dump({"frame_symbols": [], "n_assets": N_ASSETS}, fh)


def _st_frame_is_string(p, intact):
    with open(p, "w") as fh:
        json.dump({"frame_symbols": ",".join(FRAME_SYMBOLS), "n_assets": N_ASSETS}, fh)


def _st_duplicates(p, intact):
    with open(p, "w") as fh:
        json.dump({"frame_symbols": [FRAME_SYMBOLS[0]] * N_ASSETS, "n_assets": N_ASSETS}, fh)


def _st_directory(p, intact):
    os.remove(p)
    os.mkdir(p)


def _st_chmod_000(p, intact):
    os.chmod(p, 0o000)


# (name, setup, expectation, use_the_SWAPPED_pin). Every row grades with the MATCHING pin
# except the last, which is the reported repro verbatim: truncate a real provenance and pin a
# frame that genuinely swaps slots 0/1 — the combination that used to exit 0 with a paste-ready
# LEDGER row stamped "universe_pinned": true.
STATE_TABLE = [
    ("file absent",                       _st_absent,          "proceed",  False),
    ("empty file",                        _st_empty,           "unusable", False),
    ("truncated JSON",                    _st_truncated,       "unusable", False),
    ("valid JSON but a list",             _st_list,            "unusable", False),
    ("valid JSON but null",               _st_null,            "unusable", False),
    ("object missing frame_symbols",      _st_no_key,          "unusable", False),
    ("frame_symbols an empty list",       _st_empty_frame,     "unusable", False),
    ("frame_symbols a string not a list", _st_frame_is_string, "unusable", False),
    ("frame_symbols with duplicates",     _st_duplicates,      "mismatch", False),
    ("a directory at the provenance path", _st_directory,      "unusable", False),
    ("unreadable file (chmod 000)",       _st_chmod_000,       "unusable", False),
    ("truncated JSON + slot-0/1-swapping pin (the reported repro)",
     _st_truncated, "unusable", True),
]


def _restore_provenance(p, intact):
    """Put the intact provenance back, whatever the state left behind (dir / chmod 000 / ...)."""
    if os.path.isdir(p) and not os.path.islink(p):
        shutil.rmtree(p)
    elif os.path.lexists(p):
        os.chmod(p, 0o644)
        os.remove(p)
    with open(p, "wb") as fh:
        fh.write(intact)


def main():
    failures = []
    before = (os.path.getsize(FIXTURE_DB), os.path.getmtime(FIXTURE_DB))
    work = tempfile.mkdtemp(prefix="t883_universe_pin_")
    try:
        frame_a = os.path.join(work, "frame_A.json")
        _write_frame(frame_a, FRAME_SYMBOLS)

        print("[883] training a real checkpoint under FRAME_A (imprint.py — TASK_874 — emits "
              "its OWN provenance beside it) ...")
        ckpt = os.path.join(work, "ckpt_probe")
        r = _run([IMPRINT_PY, "--db", FIXTURE_DB, "--start", START, "--end", END,
                  "--bar-minutes", str(BAR_MINUTES), "--n-assets", str(N_ASSETS),
                  "--panel-frame-from-checkpoint", frame_a, "--panel-frame-source", "minute",
                  "--imprint-mapping", "weight-native", "--iters", str(TRAIN_ITERS),
                  "--batch-size", "32", "--seed", "0", "--vector-weight-rebalance",
                  "--ckpt-out", ckpt, "--tag", "t883", "--out",
                  os.path.join(work, "imprint.json")])
        if r.returncode != 0:
            raise RuntimeError(f"imprint.py training failed (exit {r.returncode}):\n"
                               f"{r.stderr[-3000:]}")
        # imprint.frame_provenance_path_for("ckpt_probe") -> frame_provenance_probe.json
        own_prov = os.path.join(work, "frame_provenance_probe.json")
        if not os.path.exists(own_prov):
            raise RuntimeError(f"expected imprint.py (TASK_874) to emit {own_prov}; the setup "
                               f"this test depends on did not hold — see stdout:\n"
                               f"{r.stdout[-2000:]}")

        base = ["--db", FIXTURE_DB, "--start", START, "--end", END,
                "--bar-minutes", str(BAR_MINUTES), "--n-assets", str(N_ASSETS),
                "--operator", "policy-argmax", "--checkpoint", ckpt,
                "--vector-weight-rebalance"]

        # ---- gate (i): unpinned REFUSES, before any panel is built ---------------------
        print("[883R] gate (i) — unpinned checkpoint-grading run must REFUSE ...")
        r = _run([EVAL_PY] + base + ["--tag", "g1", "--out-dir", work])
        ok = (r.returncode != 0
              and "REFUSING an unpinned" in r.stderr
              and ckpt in r.stderr
              and "--panel-frame-from-checkpoint" in r.stderr
              and "--allow-unpinned-universe" in r.stderr
              and not _artifacts(work, "g1"))
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (i)")
        if not ok:
            failures.append(
                "gate (i): expected a REFUSAL naming the checkpoint, the missing "
                f"--panel-frame-from-checkpoint flag and the --allow-unpinned-universe escape "
                f"hatch, and NO results artifact. returncode={r.returncode}\n"
                f"stderr tail:\n{r.stderr[-2000:]}")

        # ---- gate (ii): pin/checkpoint MISMATCH REFUSES --------------------------------
        print("[883R] gate (ii) — a pin disagreeing with the checkpoint's own provenance must "
              "REFUSE ...")
        frame_mismatch = os.path.join(work, "frame_mismatch.json")
        _write_frame(frame_mismatch, [FRAME_SYMBOLS[1], FRAME_SYMBOLS[0]] + FRAME_SYMBOLS[2:])
        r = _run([EVAL_PY] + base +
                 ["--panel-frame-from-checkpoint", frame_mismatch,
                  "--panel-frame-source", "minute", "--tag", "g2", "--out-dir", work])
        ok = (r.returncode != 0
              and "MISMATCH" in r.stderr
              and "first differing slot: 0" in r.stderr
              and FRAME_SYMBOLS[0] in r.stderr and FRAME_SYMBOLS[1] in r.stderr
              and not _artifacts(work, "g2"))
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (ii)")
        if not ok:
            failures.append(
                "gate (ii): expected a REFUSAL naming slot 0 and both symbols "
                f"{FRAME_SYMBOLS[0]!r}/{FRAME_SYMBOLS[1]!r}. returncode={r.returncode}\n"
                f"stderr tail:\n{r.stderr[-2000:]}")

        # ---- gate (iii): --allow-unpinned-universe runs to completion, self-labelled ----
        print("[883R] gate (iii) — --allow-unpinned-universe must complete and self-label ...")
        r = _run([EVAL_PY] + base + ["--allow-unpinned-universe", "--tag", "g3",
                                     "--out-dir", work])
        arts = _artifacts(work, "g3")
        doc = None
        if r.returncode == 0 and len(arts) == 1:
            with open(os.path.join(work, arts[0])) as fh:
                doc = json.load(fh)
        ok = (r.returncode == 0
              and "--allow-unpinned-universe" in r.stdout and "SCRAMBLES" in r.stdout
              and doc is not None
              and doc.get("universe_pinned") is False
              and doc.get("universe_pin", {}).get("pin_source") is None
              and doc.get("universe_pin", {}).get("certified") is False
              # D36 Q2: the caveat travels with the pasted text, naming the mechanism. Wording
              # updated by TASK_886 (the certification row D36 scoped the freeze against):
              # unpinned (mechanism "none") can never certify, so this still reads False, but
              # the sentence is 886's ("NOT CERTIFIED ... mechanism=...") not 883R's retired one.
              and "UNIVERSE NOT CERTIFIED (TASK_886)" in (doc or {}).get("ledger_row", "")
              and "mechanism='none'" in doc.get("ledger_row", ""))
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (iii)")
        if not ok:
            failures.append(
                f"gate (iii): expected exit 0, a loud warning on stdout, "
                f"universe_pinned=false + certified=false + pin_source=null in the artifact, "
                f"and the UNIVERSE NOT CERTIFIED caveat naming the mechanism INLINE in "
                f"ledger_row. returncode={r.returncode} doc={doc}\n"
                f"stdout tail:\n{r.stdout[-2000:]}")

        # ---- gate (iv): a MATCHING pin now CERTIFIES (TASK_886) -------------------------
        # UPDATED BY TASK_886 (the certification row D36 scoped its freeze against — "until the
        # certification row lands"). D36/883R's `certified`/`universe_pinned` were hardcoded
        # False for EVERY run because no emitter wrote a `param_fingerprint` into provenance, so
        # the positive claim was structurally unreachable. TASK_888/889 made `imprint.py` write
        # `param_fingerprint` at SAVE time unconditionally, so THIS scenario — a checkpoint
        # graded immediately after training, against its own freshly-written provenance — now
        # satisfies every conjunct of `_certification_verdict` (order-exact match, realized
        # panel equals requested, loaded fingerprint equals the recorded one, zero padded
        # slots) and CERTIFIES. This is the earned positive claim TASK_886 exists to introduce,
        # not a regression: a "pin matching the checkpoint's own provenance is unaffected"
        # claim is no longer true post-886 BY DESIGN, and pinning `certified: False` here would
        # be pinning the wrong law of nature.
        print("[886] gate (iv) — a pin matching the checkpoint's own provenance, graded "
              "against the SAME freshly-trained weights, now CERTIFIES ...")
        r = _run([EVAL_PY] + base +
                 ["--panel-frame-from-checkpoint", own_prov, "--panel-frame-source", "minute",
                  "--teacher-floor", "--tag", "g4", "--out-dir", work])
        arts = _artifacts(work, "g4")
        doc = None
        if r.returncode == 0 and len(arts) == 1:
            with open(os.path.join(work, arts[0])) as fh:
                doc = json.load(fh)
        keys_ok = doc is not None and set(doc.keys()) == EXPECTED_KEYS_PINNED
        ok = (r.returncode == 0 and keys_ok
              and doc.get("universe_pinned") is True
              and doc.get("universe_pin", {}).get("certified") is True
              and doc.get("universe_pin", {}).get("verified") is True
              and doc.get("universe_pin", {}).get("verified_reason") == "certified"
              and doc.get("universe_pin", {}).get("pin_source") == "checkpoint-provenance"
              and doc.get("universe_pin", {}).get("order_exact") is True
              and doc.get("universe_pin", {}).get("pin_path") is not None
              and doc.get("universe_pin", {}).get("pin_sha256") is not None
              and doc.get("universe_pin", {}).get("n_frame_symbols") == N_ASSETS
              and "UNIVERSE CERTIFIED (TASK_886)" in (doc or {}).get("ledger_row", "")
              and "'checkpoint-provenance'" in doc.get("ledger_row", ""))
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (iv)")
        if not ok:
            extra = set(doc.keys()) - EXPECTED_KEYS_PINNED if doc else None
            missing = EXPECTED_KEYS_PINNED - set(doc.keys()) if doc else None
            failures.append(
                f"gate (iv): expected exit 0, key set == pre-883 keys + "
                f"{{'universe_pinned','universe_pin'}} (extra={extra} missing={missing}), "
                f"universe_pinned=true + certified=true + verified_reason='certified' "
                f"(TASK_886: the earned positive claim), order_exact=true, pin path/sha256/"
                f"n_frame_symbols recorded, and the UNIVERSE CERTIFIED caveat naming "
                f"'checkpoint-provenance' INLINE in ledger_row. "
                f"returncode={r.returncode}\nstderr tail:\n{r.stderr[-2000:]}")

        # ---- gate (v): the checkpoint-own-provenance STATE TABLE ------------------------
        print("[883R] gate (v) — provenance state table (only a GENUINELY ABSENT file may "
              "proceed) ...")
        with open(own_prov, "rb") as fh:
            intact = fh.read()
        rows = []
        for n, (name, setup, expect, swapped_pin) in enumerate(STATE_TABLE):
            if setup is _st_chmod_000 and os.geteuid() == 0:
                rows.append((name, expect, "SKIPPED (running as root: chmod 000 is readable)"))
                continue
            tag = f"g5_{n}"
            _restore_provenance(own_prov, intact)
            setup(own_prov, intact)
            r = _run([EVAL_PY] + base +
                     ["--panel-frame-from-checkpoint",
                      frame_mismatch if swapped_pin else frame_a,
                      "--panel-frame-source", "minute", "--tag", tag, "--out-dir", work])
            arts = _artifacts(work, tag)
            doc = None
            if arts:
                with open(os.path.join(work, arts[0])) as fh:
                    doc = json.load(fh)
            if expect == "proceed":
                observed = (f"exit {r.returncode}, artifact={bool(arts)}, "
                            f"universe_pinned={doc.get('universe_pinned') if doc else None}, "
                            f"checkpoint_provenance_found="
                            f"{doc.get('universe_pin', {}).get('checkpoint_provenance_found') if doc else None}, "
                            f"order_exact="
                            f"{doc.get('universe_pin', {}).get('order_exact') if doc else None}")
                ok = (r.returncode == 0 and len(arts) == 1 and doc is not None
                      # D36: a genuinely-absent provenance proceeds, but NEVER under a
                      # positive label — this is exactly the hole-1 spelling state, and it
                      # must disclaim, not bless.
                      and doc.get("universe_pinned") is False
                      and doc["universe_pin"]["certified"] is False
                      and doc["universe_pin"]["checkpoint_provenance_found"] is False
                      and doc["universe_pin"]["order_exact"] is None)
            else:
                marker = ("PRESENT but UNUSABLE" if expect == "unusable"
                          else "universe MISMATCH")
                observed = (f"exit {r.returncode}, artifact={bool(arts)}, "
                            f"refusal={'REFUSING' in r.stderr}, marker={marker in r.stderr}")
                ok = (r.returncode != 0 and not arts and "REFUSING" in r.stderr
                      and marker in r.stderr and ckpt in r.stderr
                      and os.path.basename(own_prov) in r.stderr)
            rows.append((name, expect, ("PASS — " if ok else "FAIL — ") + observed))
            if not ok:
                failures.append(f"gate (v) [{name}]: expected {expect}; observed {observed}\n"
                                f"stderr tail:\n{r.stderr[-1200:]}")
        _restore_provenance(own_prov, intact)
        print(f"    {'state':<36} {'expected':<10} observed")
        for name, expect, observed in rows:
            print(f"    {name:<36} {expect:<10} {observed}")

        # ---- gate (vi): the EXPLICIT-SYMBOLS pin source (fix pass, BLOCKER 1) -----------
        # `build_panel` accepts an explicit, ORDER-PRESERVING `specific_symbols=args.symbols`
        # (populated by --symbols or --periods-file "assets"), which is the mechanism
        # experiments/837_walkforward/walkforward.py uses for EVERY eval job it generates —
        # against train_jax_fast.py checkpoints, which never emit a frame_provenance sibling.
        # An explicitly-named universe is a KNOWN request: it must NOT be refused — but under
        # D36 it is NEVER blessed either (universe_pinned: false / certified: false). Three
        # states, all pinned here.
        print("[883R] gate (vi) — the EXPLICIT-SYMBOLS pin source ...")
        sym_csv = ",".join(FRAME_SYMBOLS)
        # (vi-a) explicit order AGREES with the checkpoint's own provenance -> proceeds, and
        #        says so honestly: source explicit-symbols, cross-check VERIFIED, order_exact
        #        true. UPDATED BY TASK_886 (D36 scoped 883R's freeze as lasting "until the
        #        certification row lands"): this checkpoint is graded immediately after
        #        training against its OWN freshly-written provenance (param_fingerprint set
        #        unconditionally by imprint.py, TASK_888/889), so every conjunct of
        #        _certification_verdict holds and this now CERTIFIES — the mechanism split
        #        (explicit-symbols/periods-file/checkpoint-provenance) does not gate the
        #        predicate, only WHICH provenance is cross-checked.
        r = _run([EVAL_PY] + base + ["--symbols", sym_csv, "--tag", "g6a", "--out-dir", work])
        arts = _artifacts(work, "g6a")
        doc = None
        if r.returncode == 0 and len(arts) == 1:
            with open(os.path.join(work, arts[0])) as fh:
                doc = json.load(fh)
        up = (doc or {}).get("universe_pin", {})
        ok = (r.returncode == 0 and doc is not None
              and doc.get("universe_pinned") is True
              and up.get("certified") is True
              and up.get("verified") is True
              and up.get("verified_reason") == "certified"
              and up.get("pin_source") == "explicit-symbols"
              and up.get("explicit_symbols") == FRAME_SYMBOLS
              and up.get("n_frame_symbols") == N_ASSETS
              and up.get("verified_against_checkpoint_provenance") is True
              and up.get("order_exact") is True
              and "UNIVERSE CERTIFIED (TASK_886)" in (doc or {}).get("ledger_row", "")
              and "'explicit-symbols'" in doc.get("ledger_row", ""))
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (vi-a) explicit symbols AGREEING with the "
              f"checkpoint's own provenance, graded against the SAME freshly-trained weights, "
              f"now CERTIFIES (TASK_886)")
        if not ok:
            failures.append(
                f"gate (vi-a): expected exit 0 with universe_pinned=true, certified=true, "
                f"verified_reason='certified', pin_source='explicit-symbols', "
                f"verified_against_checkpoint_provenance=true, order_exact=true, and the "
                f"UNIVERSE CERTIFIED caveat naming 'explicit-symbols' INLINE in ledger_row. "
                f"returncode={r.returncode} universe_pin={up}\n"
                f"stderr tail:\n{r.stderr[-2000:]}")

        # (vi-b) explicit order DISAGREES with the checkpoint's own provenance -> REFUSE,
        #        naming the first differing slot and both symbols (same rule as the pin path).
        swapped = [FRAME_SYMBOLS[1], FRAME_SYMBOLS[0]] + FRAME_SYMBOLS[2:]
        r = _run([EVAL_PY] + base + ["--symbols", ",".join(swapped), "--tag", "g6b",
                                     "--out-dir", work])
        ok = (r.returncode != 0
              and "MISMATCH" in r.stderr
              and "first differing slot: 0" in r.stderr
              and FRAME_SYMBOLS[0] in r.stderr and FRAME_SYMBOLS[1] in r.stderr
              and not _artifacts(work, "g6b"))
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (vi-b) explicit symbols DISAGREEING with "
              f"the checkpoint's own provenance REFUSE")
        if not ok:
            failures.append(
                f"gate (vi-b): expected a REFUSAL naming slot 0 and both symbols and NO "
                f"artifact. returncode={r.returncode}\nstderr tail:\n{r.stderr[-2000:]}")

        # (vi-c) the ORDINARY train_jax_fast case: explicit order, checkpoint carries NO
        #        provenance of its own -> runs, disclaimed, and records honestly that nothing
        #        independent verified it. This is the state the pre-fix guard refused outright.
        os.remove(own_prov)
        try:
            r = _run([EVAL_PY] + base + ["--symbols", sym_csv, "--tag", "g6c",
                                         "--out-dir", work])
        finally:
            _restore_provenance(own_prov, intact)
        arts = _artifacts(work, "g6c")
        doc = None
        if r.returncode == 0 and len(arts) == 1:
            with open(os.path.join(work, arts[0])) as fh:
                doc = json.load(fh)
        up = (doc or {}).get("universe_pin", {})
        ok = (r.returncode == 0 and doc is not None
              and doc.get("universe_pinned") is False
              and up.get("certified") is False
              and up.get("pin_source") == "explicit-symbols"
              and up.get("checkpoint_provenance_found") is False
              and up.get("verified_against_checkpoint_provenance") is False
              and up.get("order_exact") is None)
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (vi-c) explicit symbols against a "
              f"provenance-less checkpoint run, disclaimed and recorded as UNVERIFIED")
        if not ok:
            failures.append(
                f"gate (vi-c): expected exit 0 with universe_pinned=false, certified=false, "
                f"pin_source='explicit-symbols', checkpoint_provenance_found=false, "
                f"verified_against_checkpoint_provenance=false, order_exact=null. "
                f"returncode={r.returncode} universe_pin={up}\nstderr tail:\n{r.stderr[-2000:]}")

        # ---- gate (vii): the REALIZED-PANEL EQUALITY ASSERTION (883R NEW, D36) ----------
        # Hole 2's mechanism: the explicit-symbols request reaches build_panel, which
        # SILENTLY DROPS a DB-absent symbol (every later slot shifts — run 29 measured 6/8
        # slots wrong and a paste-ready row at net@21 -3.953) or duplicate-pads when
        # n_symbols < --n-assets. 883R refuses BOTH after build_panel returns, naming the
        # first differing slot and both symbols. The checkpoint's own provenance is REMOVED
        # for these legs so gate (ii)'s cross-check cannot fire first and mask the assertion
        # under test (with provenance present, a wrong explicit list refuses earlier, at the
        # cross-check — same refusal, different mechanism).
        print("[883R] gate (vii) — realized panel must equal the requested universe, "
              "order-exact ...")
        os.remove(own_prov)
        try:
            # (vii-a) a DB-absent symbol in the explicit list -> REFUSE (the drop is no
            #         longer graded; slot 7 requested 'NOPE/USD', realized a shifted/pad
            #         symbol).
            absent = ",".join(FRAME_SYMBOLS[:7] + ["NOPE/USD"])
            r = _run([EVAL_PY] + base + ["--symbols", absent, "--tag", "g7a",
                                         "--out-dir", work])
            ok = (r.returncode != 0
                  and "REALIZED panel" in r.stderr
                  and "first differing slot: 7" in r.stderr
                  and "NOPE/USD" in r.stderr
                  and not _artifacts(work, "g7a"))
            print(f"    [{'PASS' if ok else 'FAIL'}] gate (vii-a) DB-absent symbol in the "
                  f"explicit list REFUSES (the run-29 hole-2 drop)")
            if not ok:
                failures.append(
                    f"gate (vii-a): expected a REFUSAL naming the realized-panel mismatch at "
                    f"slot 7 with 'NOPE/USD' and NO artifact. returncode={r.returncode}\n"
                    f"stderr tail:\n{r.stderr[-2000:]}")
            # (vii-b) n_symbols < --n-assets -> REFUSE (duplicate #pad slots are not the
            #         requested universe).
            short = ",".join(FRAME_SYMBOLS[:7])
            r = _run([EVAL_PY] + base + ["--symbols", short, "--tag", "g7b",
                                         "--out-dir", work])
            ok = (r.returncode != 0
                  and "REALIZED panel" in r.stderr
                  and "first differing slot: 7" in r.stderr
                  and "#pad7" in r.stderr
                  and not _artifacts(work, "g7b"))
            print(f"    [{'PASS' if ok else 'FAIL'}] gate (vii-b) n_symbols < --n-assets "
                  f"REFUSES (duplicate pad slots)")
            if not ok:
                failures.append(
                    f"gate (vii-b): expected a REFUSAL naming the realized-panel mismatch at "
                    f"slot 7 (requested null vs realized '#pad7') and NO artifact. "
                    f"returncode={r.returncode}\nstderr tail:\n{r.stderr[-2000:]}")
        finally:
            _restore_provenance(own_prov, intact)

        # ---- gate (viii): cycle-2 fix-pass legs ---------------------------------------
        print("[883R] gate (viii) — cycle-2 fix-pass legs ...")
        # (viii-a) DUPLICATE explicit symbols REFUSE on DISTINCTNESS — checked FIRST, before
        #          the cross-check (rationale: distinctness is a legality property of the
        #          request itself, the same rule _load_frame_provenance applies to a pin
        #          FILE; it must fire even where no provenance exists to cross-check against,
        #          and it gives the operator the accurate defect when both would fire). The
        #          duplicate here (XBT/USD twice, REP/USD dropped) would ALSO mismatch the
        #          checkpoint's provenance at slot 1 — the distinctness refusal must win.
        dup = [FRAME_SYMBOLS[0]] + FRAME_SYMBOLS[:7]      # XBT/USD at slots 0 AND 1
        r = _run([EVAL_PY] + base + ["--symbols", ",".join(dup), "--tag", "g8a",
                                     "--out-dir", work])
        ok = (r.returncode != 0
              and "REFUSING a DUPLICATE" in r.stderr
              and f"'{FRAME_SYMBOLS[0]}'" in r.stderr
              and "slots [0, 1]" in r.stderr
              and not _artifacts(work, "g8a"))
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (viii-a) duplicate explicit symbols "
              f"REFUSE on distinctness (naming symbol + slots; cross-check would also fire)")
        if not ok:
            failures.append(
                f"gate (viii-a): expected a distinctness REFUSAL naming {FRAME_SYMBOLS[0]!r} "
                f"and slots [0, 1], NO artifact. returncode={r.returncode}\n"
                f"stderr tail:\n{r.stderr[-2000:]}")
        # (viii-b) the SAME duplicate against a PROVENANCE-LESS checkpoint (the ordinary
        #          train_jax_fast case — no cross-check exists to catch anything) still
        #          refuses on distinctness.
        os.remove(own_prov)
        try:
            r = _run([EVAL_PY] + base + ["--symbols", ",".join(dup), "--tag", "g8b",
                                         "--out-dir", work])
            ok = (r.returncode != 0
                  and "REFUSING a DUPLICATE" in r.stderr
                  and "slots [0, 1]" in r.stderr
                  and not _artifacts(work, "g8b"))
        finally:
            _restore_provenance(own_prov, intact)
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (viii-b) duplicate explicit symbols "
              f"REFUSE against a provenance-less checkpoint")
        if not ok:
            failures.append(
                f"gate (viii-b): expected the same distinctness REFUSAL with NO artifact. "
                f"returncode={r.returncode}\nstderr tail:\n{r.stderr[-2000:]}")
        # (viii-c) a FIFO at the auto-derived checkpoint-provenance path refuses FAST (the
        #          pre-fix reader blocked on open() forever — reproduced 45s, killed). The
        #          60s subprocess timeout is the hang tripwire: a pre-fix module dies HERE
        #          with TimeoutExpired, not with an assertion diff.
        os.remove(own_prov)
        os.mkfifo(own_prov)
        try:
            r = _run([EVAL_PY] + base +
                     ["--panel-frame-from-checkpoint", frame_a,
                      "--panel-frame-source", "minute", "--tag", "g8c", "--out-dir", work],
                     timeout=60)
            ok = (r.returncode != 0
                  and "PRESENT but UNUSABLE" in r.stderr
                  and "NOT a regular file" in r.stderr
                  and not _artifacts(work, "g8c"))
        finally:
            _restore_provenance(own_prov, intact)
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (viii-c) FIFO at the provenance path "
              f"refuses fast (no hang)")
        if not ok:
            failures.append(
                f"gate (viii-c): expected a FAST present-but-unusable REFUSAL naming the "
                f"non-regular-file state, NO artifact. returncode={r.returncode}\n"
                f"stderr tail:\n{r.stderr[-2000:]}")
        # (viii-d) refusal text echoes NO file content: key NAMES stay out (an object with no
        #          frame_symbols reports a key COUNT), entry VALUES stay out (a non-string
        #          frame entry reports only its slot index).
        with open(own_prov, "w") as fh:
            json.dump({"api_key": "s3cr3t-key-name-probe", "n_assets": N_ASSETS}, fh)
        try:
            r = _run([EVAL_PY] + base +
                     ["--panel-frame-from-checkpoint", frame_a,
                      "--panel-frame-source", "minute", "--tag", "g8d1", "--out-dir", work])
            ok = (r.returncode != 0
                  and "PRESENT but UNUSABLE" in r.stderr
                  and "api_key" not in r.stderr
                  and "s3cr3t-key-name-probe" not in r.stderr
                  and "top-level keys present" in r.stderr)
        finally:
            _restore_provenance(own_prov, intact)
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (viii-d1) key NAMES not echoed")
        if not ok:
            failures.append(
                f"gate (viii-d1): expected the refusal to report a key COUNT, not names. "
                f"returncode={r.returncode}\nstderr tail:\n{r.stderr[-2000:]}")
        bad_frame = list(FRAME_SYMBOLS)
        bad_frame[1] = {"nested": "s3cr3t-entry-value-probe"}
        with open(own_prov, "w") as fh:
            json.dump({"frame_symbols": bad_frame, "n_assets": N_ASSETS}, fh)
        try:
            r = _run([EVAL_PY] + base +
                     ["--panel-frame-from-checkpoint", frame_a,
                      "--panel-frame-source", "minute", "--tag", "g8d2", "--out-dir", work])
            ok = (r.returncode != 0
                  and "PRESENT but UNUSABLE" in r.stderr
                  and "slot(s) [1]" in r.stderr
                  and "s3cr3t-entry-value-probe" not in r.stderr
                  and "nested" not in r.stderr)
        finally:
            _restore_provenance(own_prov, intact)
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (viii-d2) entry VALUES not echoed")
        if not ok:
            failures.append(
                f"gate (viii-d2): expected the refusal to name slot [1] only, not the entry "
                f"value. returncode={r.returncode}\nstderr tail:\n{r.stderr[-2000:]}")
        # (viii-e) an EMPTY --panel-frame-from-checkpoint is reported as GIVEN-BUT-EMPTY,
        #          not mis-reported as a missing flag.
        r = _run([EVAL_PY] + base + ["--panel-frame-from-checkpoint", "",
                                     "--tag", "g8e", "--out-dir", work])
        ok = (r.returncode != 0
              and "EMPTY path" in r.stderr
              and "missing flag" not in r.stderr
              and not _artifacts(work, "g8e"))
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (viii-e) empty pin path reported as "
              f"given-but-empty, not as a missing flag")
        if not ok:
            failures.append(
                f"gate (viii-e): expected a given-but-EMPTY-path REFUSAL, not the "
                f"missing-flag one. returncode={r.returncode}\n"
                f"stderr tail:\n{r.stderr[-2000:]}")

        # ---- gate (ix): cycle-3 — distinctness on the LOADERS' identity function --------
        # normalize_symbol_key (upper + strip '/') is what every panel loader resolves
        # through, so XBT/USD, xbt/usd and XBTUSD all load the SAME asset. Exact-string
        # distinctness was blind to that slot collapse (cycle-3 judge: both slots carried
        # asset_id 1091 under a passing run). Both duplicate sentinels — the explicit-list
        # check in enforce_universe_pin and _load_frame_provenance — now judge the
        # normalized key.
        print("[883R] gate (ix) — variant-spelling duplicates (loaders' identity function) ...")
        # (ix-a) CASE-variant duplicate via --symbols REFUSES, naming BOTH spellings.
        case_dup = ["XBT/USD", "xbt/usd"] + FRAME_SYMBOLS[2:]
        r = _run([EVAL_PY] + base + ["--symbols", ",".join(case_dup), "--tag", "g9a",
                                     "--out-dir", work])
        ok = (r.returncode != 0
              and "REFUSING a DUPLICATE" in r.stderr
              and "'XBT/USD'" in r.stderr and "'xbt/usd'" in r.stderr
              and "slots [0, 1]" in r.stderr
              and not _artifacts(work, "g9a"))
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (ix-a) case-variant duplicate via "
              f"--symbols REFUSES (both spellings named)")
        if not ok:
            failures.append(
                f"gate (ix-a): expected a distinctness REFUSAL naming 'XBT/USD' and "
                f"'xbt/usd' at slots [0, 1], NO artifact. returncode={r.returncode}\n"
                f"stderr tail:\n{r.stderr[-2000:]}")
        # (ix-b) SLASH-variant duplicate via --symbols REFUSES.
        slash_dup = ["XBT/USD", "XBTUSD"] + FRAME_SYMBOLS[2:]
        r = _run([EVAL_PY] + base + ["--symbols", ",".join(slash_dup), "--tag", "g9b",
                                     "--out-dir", work])
        ok = (r.returncode != 0
              and "REFUSING a DUPLICATE" in r.stderr
              and "'XBT/USD'" in r.stderr and "'XBTUSD'" in r.stderr
              and "slots [0, 1]" in r.stderr
              and not _artifacts(work, "g9b"))
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (ix-b) slash-variant duplicate via "
              f"--symbols REFUSES")
        if not ok:
            failures.append(
                f"gate (ix-b): expected a distinctness REFUSAL naming 'XBT/USD' and "
                f"'XBTUSD' at slots [0, 1], NO artifact. returncode={r.returncode}\n"
                f"stderr tail:\n{r.stderr[-2000:]}")
        # (ix-c) a variant-duplicate pin FILE refuses at _load_frame_provenance (the pin
        #        path's own sentinel, "DUPLICATE symbol(s)"). The checkpoint's own
        #        provenance is REMOVED so gate (ii)'s exact-string cross-check cannot fire
        #        first and mask the sentinel under test.
        os.remove(own_prov)
        frame_vardup = os.path.join(work, "frame_vardup.json")
        _write_frame(frame_vardup, ["XBT/USD", "xbt/usd"] + FRAME_SYMBOLS[2:])
        try:
            r = _run([EVAL_PY] + base +
                     ["--panel-frame-from-checkpoint", frame_vardup,
                      "--panel-frame-source", "minute", "--tag", "g9c", "--out-dir", work])
            ok = (r.returncode != 0
                  and "DUPLICATE symbol(s)" in r.stderr
                  and "'XBT/USD'" in r.stderr and "'xbt/usd'" in r.stderr
                  and "slots [0, 1]" in r.stderr
                  and not _artifacts(work, "g9c"))
        finally:
            _restore_provenance(own_prov, intact)
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (ix-c) variant-duplicate pin FILE "
              f"REFUSES at _load_frame_provenance")
        if not ok:
            failures.append(
                f"gate (ix-c): expected the pin-file DUPLICATE refusal naming both "
                f"spellings at slots [0, 1], NO artifact. returncode={r.returncode}\n"
                f"stderr tail:\n{r.stderr[-2000:]}")
        # (ix-d) a COLLUDING provenance carrying the SAME variant frame (the exact-string
        #        cross-check passes on it) still refuses on normalized distinctness.
        with open(own_prov, "w") as fh:
            json.dump({"frame_symbols": case_dup, "n_assets": N_ASSETS}, fh)
        try:
            r = _run([EVAL_PY] + base + ["--symbols", ",".join(case_dup), "--tag", "g9d",
                                         "--out-dir", work])
            ok = (r.returncode != 0
                  and "REFUSING a DUPLICATE" in r.stderr
                  and "slots [0, 1]" in r.stderr
                  and not _artifacts(work, "g9d"))
        finally:
            _restore_provenance(own_prov, intact)
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (ix-d) colluding provenance (same "
              f"variants) still REFUSES on distinctness")
        if not ok:
            failures.append(
                f"gate (ix-d): expected the distinctness REFUSAL even with a colluding "
                f"provenance. returncode={r.returncode}\nstderr tail:\n{r.stderr[-2000:]}")
        # (ix-e) a VALID duplicate-free run with mixed-case spellings that normalize to
        #        DISTINCT keys still completes (the loaders keep the requested spelling, so
        #        the realized-panel assertion sees exactly the request).
        os.remove(own_prov)
        lower = [s.lower() for s in FRAME_SYMBOLS]
        try:
            r = _run([EVAL_PY] + base + ["--symbols", ",".join(lower), "--tag", "g9e",
                                         "--out-dir", work])
            arts = _artifacts(work, "g9e")
            doc = None
            if r.returncode == 0 and len(arts) == 1:
                with open(os.path.join(work, arts[0])) as fh:
                    doc = json.load(fh)
            ok = (r.returncode == 0 and doc is not None
                  and doc.get("universe_pinned") is False
                  and doc.get("universe_pin", {}).get("certified") is False
                  and doc.get("universe_pin", {}).get("explicit_symbols") == lower)
        finally:
            _restore_provenance(own_prov, intact)
        print(f"    [{'PASS' if ok else 'FAIL'}] gate (ix-e) mixed-case but DISTINCT request "
              f"still completes (disclaimed)")
        if not ok:
            failures.append(
                f"gate (ix-e): expected exit 0, artifact stamped universe_pinned=false and "
                f"recording the requested spellings verbatim. returncode={r.returncode}\n"
                f"stderr tail:\n{r.stderr[-2000:]}")

    finally:
        shutil.rmtree(work, ignore_errors=True)

    after = (os.path.getsize(FIXTURE_DB), os.path.getmtime(FIXTURE_DB))
    if before != after:
        failures.append(f"fixtures/trading_fixture.db changed size/mtime during this run "
                        f"({before} -> {after}) — INVARIANT VIOLATION, never regenerate it.")
    else:
        print(f"[883] fixture unchanged: size={before[0]} mtime={before[1]}")

    print("=" * 84)
    if failures:
        print(f"[883R] FAIL — {len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"[883R] PASS — gates (i)-(iv) + the {len(STATE_TABLE)}-row provenance state table "
          f"+ the 3 explicit-symbols pin-source states + the 2 realized-panel assertion "
          f"states + the 6 cycle-2 fix-pass legs + the 5 cycle-3 normalized-distinctness "
          f"legs (+ fixture-unchanged) green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
