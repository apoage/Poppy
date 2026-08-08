#!/usr/bin/env python3
"""TASK_882 — regression test pinning cross-instrument agreement semantics.

Context (LEDGER `882/instrument-disagreement`): `imprint.py`'s sanity replay and
`eval_vector.py --operator policy-argmax` both claim to replay the per-asset argmax of the
SAME checkpoint, and were found to disagree by 1.6 net@21 on the real panel (+0.8394 vs
-0.6145) — opposite economic outcomes on identical weights. Root-caused on the fixture (this
task): the two invocations were NOT given the same `--panel-frame-from-checkpoint` pin. Slot k
IS asset k positionally in the encoder (27-feat x n_assets flattened kernel), so a checkpoint
graded on a DIFFERENT universe/order than it was trained on feeds every slot's learned weights
the WRONG asset's price history. Independently confirmed against the real read-only source of
record (`trading_data.db`, 2026-08-08): the checkpoint's recorded training frame
(`runs/832_e5/frame_provenance_imprint.json`) and today's unpinned coverage-ranked universe
over the same window/n_assets disagree on 63/64 positions (24 symbols absent entirely) — the
858 backfill moved the ranking after the frame was recorded. The LEADING hypothesis in the
task brief (a `position_size_pct` sizing mismatch: imprint's checkpoints record 0.0078125 =
1/128 while eval_vector defaults to 0.1) is TESTED HERE and ELIMINATED for this failure: under
`--vector-weight-rebalance` (the config both real invocations used), `step_vector` overwrites
cash/positions wholesale and sizes limit legs at `w_unit*PV` — `position_size_pct` is DEAD CODE
(jax_trading_env.py:1791-1798,1785,1561,1418) — confirmed empirically below across an 8x spread
(0.0625 vs 0.5) producing BIT-IDENTICAL results, with a control showing the SAME spread moves
the result the moment `--vector-weight-rebalance` is dropped (proving the test is sensitive,
not just insensitive to everything).

FOUR pins, each run against a checkpoint FRESH-TRAINED on the fixture (hermetic, no tracked
baseline is read or written; `fixtures/trading_fixture.db` is opened read-only and never
regenerated — verify size/mtime unchanged before/after any run of this file):

  CHECK 1 (the agreed semantics) — FRAME-MATCHED AGREEMENT.
    Given the SAME checkpoint, SAME window, SAME `--panel-frame-from-checkpoint` pin and SAME
    `--vector-weight-rebalance` on both sides, `imprint.py`'s embedded sanity replay and
    `eval_vector.py --operator policy-argmax` MUST agree on net@21 and final_pv to
    `imprint.TV_WALK_SCORE_REL_TOL` (1e-4 relative — the pre-existing, documented float32
    reduction-order residual between imprint's unbatched walk and eval_vector's
    `jax.vmap(compute_portfolio_value)` batched walk). This is the invariant that was silently
    violated in production; this check is what should have caught it.

  CHECK 2 (the mechanism, pinned as current-behavior) — FRAME-MISMATCH DIVERGENCE.
    The SAME two invocations, but eval_vector's pin points at a DELIBERATE reversed permutation
    of the SAME 8 symbols (same set, different slot order — modelling a re-ranking event like
    the real 858 backfill). Net@21 MUST diverge beyond the CHECK-1 tolerance. This documents
    that the failure mode is real, mechanical and reproducible, not a fluke of the specific real
    run — and it will force a conscious update to this file if the harnesses are ever changed
    to reconcile frames automatically (a design change out of this task's scope).

  CHECK 3 (the eliminated hypothesis, pinned) — POSITION_SIZE_PCT INERT UNDER WEIGHT-REBALANCE.
    Frame held MATCHED (no confound from Check 2's mechanism); `--position-size-pct` swept
    0.0625 -> 0.5 (8x) on the eval_vector side while `--vector-weight-rebalance` stays ON on
    both sides. Net@21 MUST be bit-identical across the sweep. A control leg (same sweep,
    `--vector-weight-rebalance` OFF) MUST show net@21 actually move — proving Check 3's
    equality is a real inertness finding, not a test that can't detect anything.

  CHECK 4 (the LITERAL incident) — PIN OMITTED ENTIRELY.
    CHECK 2 pins "pinned-but-permuted", which is NOT what happened in production. The real
    `882/instrument-disagreement` defect was an OMITTED `--panel-frame-from-checkpoint`,
    which routes through a DIFFERENT code path: `build_panel`'s unpinned, coverage-ranked
    branch (`eval_vector.py:3282-3286`) instead of `_build_panel_framed_minute`. So this
    check drops the flag altogether against the SAME pin-trained checkpoint and asserts
    BOTH halves of the silent-drift signature:
      (a) net@21 does NOT match the imprint reference (the universe the net was trained on
          is not the universe it is being graded on), and
      (b) the artifact carries NO `panel_frame` key at all — absent, not present-but-wrong.
          (b) is the load-bearing half: it is exactly why the defect was silent. A reader of
          an unpinned artifact sees a clean `STAGE-4 EVAL` banner and a paste-ready LEDGER
          row with nothing to indicate the universe was never checked against the
          checkpoint. If (b) ever flips — e.g. the recommended `panel_frame_pinned: bool`
          stamp lands (TASK_882 "Recommended follow-up row") — this check must be UPDATED
          deliberately, and its failure is the reminder to do so.

Mutation-checked manually (report in the TASK_882 writeup): breaking CHECK 1 by pointing
eval_vector at the mismatched frame reproduces exactly the failure this file exists to catch
(red); restoring the matched pin recovers green. CHECK 4 mutation-checked by inverting each
of its two assertions independently.

CPU / fixture only; no GPU; never touches `trading_data.db`. Run:
    python experiments/882_instrument_reconciliation/test_instrument_agreement.py
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

FIXTURE_DB = os.path.join(_REPO, "fixtures", "trading_fixture.db")
IMPRINT_PY = os.path.join(_REPO, "experiments", "832_linear_imprint", "imprint.py")
EVAL_PY = os.path.join(_REPO, "experiments", "835_stage4_eval", "eval_vector.py")

# The fixture's own 8-symbol minute-slice universe (eval_vector.SMOKE_SYMBOLS / imprint's
# fixture default n_assets=8) — reused, not re-derived. This IS the fixture's natural
# coverage-ranked top-8 for the window below (verified separately); Check 2 permutes it.
FRAME_SYMBOLS = ["XBT/USD", "ETH/USD", "LTC/USD", "XMR/USD",
                 "ZEC/USD", "ETC/USD", "DASH/USD", "REP/USD"]
START, END, BAR_MINUTES, N_ASSETS = "2025-09-14", "2025-12-12", 240, 8
TRAIN_ITERS = 40                                   # small: a real checkpoint, not a good one.
                                                    # (40, not fewer: at 15 iters the per-slot
                                                    # weights are not yet differentiated enough
                                                    # for CHECK 2's frame-swap to separate from
                                                    # noise -- measured net@21 gap 0.14 at 15
                                                    # iters vs ~4.3 at 40; the mechanism is the
                                                    # same, only the margin changes.)
REL_TOL = 1e-4                                     # imprint.py TV_WALK_SCORE_REL_TOL, mirrored


def _write_frame(path, symbols, tag):
    with open(path, "w") as fh:
        json.dump({
            "frame_symbols": list(symbols), "n_assets": len(symbols),
            "source": f"TASK_882 test_instrument_agreement.py ({tag})",
            "train_window": [START, END], "db": FIXTURE_DB,
        }, fh)


def _run(cmd, label):
    print(f"    $ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=_REPO, capture_output=True, text=True, timeout=280)
    if r.returncode != 0:
        raise RuntimeError(
            f"{label} exited {r.returncode}\n--- stdout tail ---\n{r.stdout[-3000:]}\n"
            f"--- stderr tail ---\n{r.stderr[-3000:]}")
    return r


def train_checkpoint(work, frame_path, ckpt_out, out_json, position_size_pct=None,
                      weight_rebalance=True, seed=0):
    cmd = [sys.executable, IMPRINT_PY,
           "--db", FIXTURE_DB, "--start", START, "--end", END,
           "--bar-minutes", str(BAR_MINUTES), "--n-assets", str(N_ASSETS),
           "--panel-frame-from-checkpoint", frame_path, "--panel-frame-source", "minute",
           "--imprint-mapping", "weight-native",
           "--iters", str(TRAIN_ITERS), "--batch-size", "32", "--seed", str(seed),
           "--ckpt-out", ckpt_out, "--tag", "t882", "--out", out_json]
    if weight_rebalance:
        cmd.append("--vector-weight-rebalance")
    if position_size_pct is not None:
        cmd += ["--position-size-pct", str(position_size_pct)]
    _run(cmd, "imprint.py (train + sanity replay)")
    with open(out_json) as fh:
        return json.load(fh)


def eval_checkpoint(work, frame_path, ckpt_dir, out_dir, tag, position_size_pct=None,
                     weight_rebalance=True):
    """``frame_path=None`` OMITS --panel-frame-from-checkpoint entirely (CHECK 4's path:
    build_panel's unpinned coverage-ranked branch), rather than passing a different pin."""
    cmd = [sys.executable, EVAL_PY,
           "--db", FIXTURE_DB, "--start", START, "--end", END,
           "--bar-minutes", str(BAR_MINUTES), "--n-assets", str(N_ASSETS),
           "--operator", "policy-argmax", "--checkpoint", ckpt_dir,
           "--tag", tag, "--out-dir", out_dir]
    if frame_path is not None:
        cmd += ["--panel-frame-from-checkpoint", frame_path,
                "--panel-frame-source", "minute"]
    if weight_rebalance:
        cmd += ["--vector-weight-rebalance", "--teacher-floor"]
    if position_size_pct is not None:
        cmd += ["--position-size-pct", str(position_size_pct)]
    _run(cmd, "eval_vector.py (--operator policy-argmax)")
    # eval_vector names its output <out_dir>/results_<tag>_<ckpt-label>.json
    matches = [f for f in os.listdir(out_dir) if f.startswith(f"results_{tag}_") and f.endswith(".json")]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one eval_vector artifact for tag={tag!r}, found {matches}")
    with open(os.path.join(out_dir, matches[0])) as fh:
        return json.load(fh)


def rel_or_abs_close(a, b, rtol=REL_TOL, atol=1e-6):
    return abs(a - b) <= max(atol, rtol * max(abs(a), abs(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="keep the tmp workdir (debugging)")
    args = ap.parse_args()

    before = (os.path.getsize(FIXTURE_DB), os.path.getmtime(FIXTURE_DB))

    work = tempfile.mkdtemp(prefix="t882_instrument_agreement_")
    failures = []
    try:
        frame_a = os.path.join(work, "frame_A.json")
        frame_b = os.path.join(work, "frame_B_reversed.json")
        _write_frame(frame_a, FRAME_SYMBOLS, "matched / natural order")
        _write_frame(frame_b, list(reversed(FRAME_SYMBOLS)), "MISMATCHED / reversed permutation")

        print("[882] training the checkpoint under FRAME_A (weight-rebalance ON, position_size_pct 0.0625) ...")
        ckpt = os.path.join(work, "ckpt")
        imprint_out = train_checkpoint(work, frame_a, ckpt, os.path.join(work, "imprint_A.json"))
        net21_imprint = imprint_out["sanity"]["per_tier"]["21"]["policy_inenv"]
        pv_imprint = imprint_out["sanity"]["per_tier"]["21"]["final_pv"]
        print(f"    imprint.py sanity replay: net@21={net21_imprint!r} final_pv={pv_imprint!r}")

        # ---- CHECK 1: frame-matched agreement -----------------------------------------
        print("[882] CHECK 1 -- eval_vector.py under the MATCHED frame (must agree) ...")
        eval_a = eval_checkpoint(work, frame_a, ckpt, work, "evalA")
        net21_eval_a = eval_a["tiers"]["21"]["sharpe_full"]
        pv_eval_a = eval_a["tiers"]["21"]["final_pv"]
        print(f"    eval_vector.py policy-argmax: net@21={net21_eval_a!r} final_pv={pv_eval_a!r}")
        ok1_sharpe = rel_or_abs_close(net21_imprint, net21_eval_a)
        ok1_pv = rel_or_abs_close(pv_imprint, pv_eval_a, rtol=1e-6, atol=0.01)
        status1 = "PASS" if (ok1_sharpe and ok1_pv) else "FAIL"
        print(f"    [{status1}] CHECK 1 frame-matched agreement "
              f"(sharpe rel-close={ok1_sharpe}, pv close={ok1_pv})")
        if status1 == "FAIL":
            failures.append("CHECK 1 (frame-matched agreement) — imprint.py and eval_vector.py "
                            f"disagree under an IDENTICAL frame pin: net@21 {net21_imprint!r} vs "
                            f"{net21_eval_a!r}, final_pv {pv_imprint!r} vs {pv_eval_a!r}. This is "
                            f"the invariant TASK_882 exists to restore — instruments must agree "
                            f"when given equivalent configuration.")

        # ---- CHECK 2: frame-mismatch divergence (pins the mechanism) ------------------
        print("[882] CHECK 2 -- eval_vector.py under the MISMATCHED (reversed) frame (must diverge) ...")
        eval_b = eval_checkpoint(work, frame_b, ckpt, work, "evalB")
        net21_eval_b = eval_b["tiers"]["21"]["sharpe_full"]
        print(f"    eval_vector.py policy-argmax (mismatched frame): net@21={net21_eval_b!r}")
        diverges = not rel_or_abs_close(net21_imprint, net21_eval_b, rtol=0.05, atol=0.3)
        status2 = "PASS" if diverges else "FAIL"
        print(f"    [{status2}] CHECK 2 frame-mismatch divergence (diverges={diverges})")
        if status2 == "FAIL":
            failures.append("CHECK 2 (frame-mismatch divergence) — a checkpoint graded under a "
                             "REVERSED slot order matched the correctly-framed reference within "
                             "tolerance; the slot-identity mechanism this file pins no longer "
                             "reproduces on the fixture (re-derive the mechanism before trusting "
                             "CHECK 1's pass).")

        # ---- CHECK 3: position_size_pct inert under weight-rebalance ------------------
        print("[882] CHECK 3 -- position_size_pct sweep under FRAME_A, --vector-weight-rebalance ON ...")
        eval_a_pp05 = eval_checkpoint(work, frame_a, ckpt, work, "evalA_pp05", position_size_pct=0.5)
        net21_pp05_on = eval_a_pp05["tiers"]["21"]["sharpe_full"]
        print(f"    weight-rebalance ON,  position_size_pct=0.5   : net@21={net21_pp05_on!r}")
        inert = rel_or_abs_close(net21_eval_a, net21_pp05_on, rtol=1e-6, atol=1e-6)
        print(f"    weight-rebalance ON,  position_size_pct=0.1(*): net@21={net21_eval_a!r}  (* eval_vector default)")

        print("[882] CHECK 3 control -- SAME sweep with --vector-weight-rebalance OFF (must move) ...")
        eval_a_wroff_pp01 = eval_checkpoint(work, frame_a, ckpt, work, "evalA_wroff_pp01",
                                            position_size_pct=0.1, weight_rebalance=False)
        eval_a_wroff_pp05 = eval_checkpoint(work, frame_a, ckpt, work, "evalA_wroff_pp05",
                                            position_size_pct=0.5, weight_rebalance=False)
        net21_wroff_01 = eval_a_wroff_pp01["tiers"]["21"].get("sharpe_full")
        net21_wroff_05 = eval_a_wroff_pp05["tiers"]["21"].get("sharpe_full")
        print(f"    weight-rebalance OFF, position_size_pct=0.1   : net@21={net21_wroff_01!r}")
        print(f"    weight-rebalance OFF, position_size_pct=0.5   : net@21={net21_wroff_05!r}")
        control_moves = (net21_wroff_01 is None or net21_wroff_05 is None or
                          not rel_or_abs_close(net21_wroff_01, net21_wroff_05, rtol=1e-6, atol=1e-6))
        status3 = "PASS" if (inert and control_moves) else "FAIL"
        print(f"    [{status3}] CHECK 3 position_size_pct inert under weight-rebalance "
              f"(inert={inert}, control_moves={control_moves})")
        if status3 == "FAIL":
            failures.append("CHECK 3 (position_size_pct inert under weight-rebalance) — either "
                             "an 8x position_size_pct spread moved net@21 while "
                             "--vector-weight-rebalance was ON (the dead-code claim no longer "
                             "holds — re-open the leading hypothesis this task eliminated), or "
                             "the OFF control failed to move at all (the test cannot detect "
                             "sizing changes and CHECK 3's PASS would be meaningless).")

        # ---- CHECK 4: the pin OMITTED entirely (the literal production incident) --------
        print("[882] CHECK 4 -- eval_vector.py with NO --panel-frame-from-checkpoint at all ...")
        eval_unpinned = eval_checkpoint(work, None, ckpt, work, "evalUnpinned")
        net21_unpinned = eval_unpinned["tiers"]["21"].get("sharpe_full")
        pf_absent = "panel_frame" not in eval_unpinned
        print(f"    eval_vector.py policy-argmax (NO pin): net@21={net21_unpinned!r} "
              f"panel_frame_key_present={not pf_absent}")
        unpinned_diverges = (net21_unpinned is None or
                             not rel_or_abs_close(net21_imprint, net21_unpinned,
                                                  rtol=0.05, atol=0.3))
        status4 = "PASS" if (unpinned_diverges and pf_absent) else "FAIL"
        print(f"    [{status4}] CHECK 4 unpinned run drifts silently "
              f"(diverges={unpinned_diverges}, panel_frame_absent={pf_absent})")
        if status4 == "FAIL":
            failures.append(
                "CHECK 4 (pin OMITTED entirely) — either an UNPINNED eval_vector run "
                f"reproduced the frame-matched reference within tolerance (net@21 "
                f"{net21_unpinned!r} vs imprint {net21_imprint!r}: the unpinned "
                "coverage-ranked universe happens to coincide with the checkpoint's frame on "
                "this fixture, so this check no longer exercises the silent-drift path — "
                "re-derive the fixture universe before trusting it), or the unpinned artifact "
                "GREW a `panel_frame` key (the fail-open asymmetry this check documents has "
                "been fixed or changed — update this check deliberately and revisit the "
                "TASK_882 recommended follow-up row on `panel_frame_pinned`).")

    finally:
        if args.keep:
            print(f"[882] workdir kept: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)

    after = (os.path.getsize(FIXTURE_DB), os.path.getmtime(FIXTURE_DB))
    if before != after:
        failures.append(f"fixtures/trading_fixture.db changed size/mtime during this run "
                         f"({before} -> {after}) — INVARIANT VIOLATION, never regenerate it.")
    else:
        print(f"[882] fixture unchanged: size={before[0]} mtime={before[1]}")

    print("=" * 84)
    if failures:
        print(f"[882] FAIL — {len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[882] PASS — all 4 checks (+ fixture-unchanged) green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
