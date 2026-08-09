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

FOUR pins (plus CHECK 2b, the in-process premise leg), each run against a checkpoint
FRESH-TRAINED on the fixture (hermetic, no tracked
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

  CHECK 2 (the mechanism) — FRAME-MISMATCH now REFUSES (TASK_883 gate ii).
    The SAME two invocations, but eval_vector's pin points at a DELIBERATE reversed permutation
    of the SAME 8 symbols (same set, different slot order — modelling a re-ranking event like
    the real 858 backfill). UPDATED for TASK_883: imprint.py (TASK_874) now emits its OWN frame
    provenance beside the checkpoint it writes, and eval_vector.py (TASK_883) now cross-checks
    a given pin against that provenance and REFUSES on disagreement — order-exact, naming the
    first differing slot and both symbols — rather than silently producing a divergent number.
    This is a STRICTLY BETTER outcome than this check's original pre-883 assertion (silent
    divergence): the exact "pinned-but-permuted" case this check exists to exercise is now
    caught at the source instead of merely reproduced. This check now pins THAT: the mismatched
    invocation must exit non-zero and its stderr must name the mismatch mechanism.

  ORIGINAL CHECK 2 (pre-883, retired as a SUBPROCESS leg) — the SAME setup used to be graded
    and used to diverge silently by design (net@21 MUST diverge beyond the CHECK-1 tolerance);
    this is why the invariant was silently violated in production and is exactly the gap
    TASK_883 closes. Its MEASUREMENT is not retired — see CHECK 2b.

  CHECK 2b (the premise, still measured) — POSITIONAL SENSITIVITY, IN-PROCESS.
    CHECK 2's refusal is the right outcome, but it fires BEFORE the model ever sees the
    mismatched frame — so on its own it can no longer prove the premise BOTH 882 and 883 rest
    on: that the encoder is positionally universe-locked, i.e. that a permuted frame really
    does change the number (if it did not, the whole 882 root-cause and the 883 guard would be
    ceremony). This check restores that proof by grading the SAME checkpoint on the matched and
    the reversed frame through the IN-PROCESS path — `build_panel` + `evaluate_checkpoint`
    called directly, which does not run `main()`'s TASK_883 guard and therefore does not weaken
    it. Two legs: (a) the in-process matched-frame read must reproduce CHECK 1's SUBPROCESS
    net@21 (proving the instrument is the same computation, not a private one), and (b) the
    reversed frame must then diverge beyond the retired CHECK 2's tolerance.

  CHECK 3 (the eliminated hypothesis, pinned) — POSITION_SIZE_PCT INERT UNDER WEIGHT-REBALANCE.
    Frame held MATCHED (no confound from Check 2's mechanism); `--position-size-pct` swept
    0.0625 -> 0.5 (8x) on the eval_vector side while `--vector-weight-rebalance` stays ON on
    both sides. Net@21 MUST be bit-identical across the sweep. A control leg (same sweep,
    `--vector-weight-rebalance` OFF) MUST show net@21 actually move — proving Check 3's
    equality is a real inertness finding, not a test that can't detect anything.

  CHECK 4 (the LITERAL incident) — PIN OMITTED ENTIRELY, now REFUSES (TASK_883 gate i).
    CHECK 2 pins "pinned-but-permuted", which is NOT what happened in production. The real
    `882/instrument-disagreement` defect was an OMITTED `--panel-frame-from-checkpoint`, which
    used to route through a DIFFERENT code path: `build_panel`'s unpinned, coverage-ranked
    branch (`eval_vector.py:3282-3286`) instead of `_build_panel_framed_minute`. UPDATED for
    TASK_883: this is the exact `panel_frame_pinned`-style follow-up this docstring predicted
    ("If (b) ever flips ... this check must be UPDATED deliberately, and its failure is the
    reminder to do so") — it flipped, harder than proposed: an unpinned checkpoint-grading run
    now REFUSES outright, before any panel is built, rather than completing with an absent
    `panel_frame` key. This check now pins THAT: dropping the flag against the SAME
    pin-trained checkpoint must exit non-zero and name the missing flag.

  ORIGINAL CHECK 4 (pre-883, retired) — the SAME setup used to complete and silently drift:
    (a) net@21 did NOT match the imprint reference, and (b) the artifact carried NO
    `panel_frame` key at all — absent, not present-but-wrong. (b) was the load-bearing half: a
    reader of an unpinned artifact saw a clean `STAGE-4 EVAL` banner and a paste-ready LEDGER
    row with nothing to indicate the universe was never checked against the checkpoint. This is
    exactly the silent-drift signature TASK_883 exists to close.

Mutation-checked manually (report in the TASK_882 writeup): breaking CHECK 1 by pointing
eval_vector at the mismatched frame reproduces exactly the failure this file exists to catch
(red); restoring the matched pin recovers green. CHECKS 2/4 (TASK_883 update) mutation-checked
by temporarily reverting to the pre-883 assertions above against the current module: both fail
red (a REFUSAL where a divergent-but-successful run used to be expected), confirming these
checks now exercise the NEW behavior and would catch its regression.

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


def _eval_cmd(frame_path, ckpt_dir, out_dir, tag, position_size_pct=None, weight_rebalance=True):
    """``frame_path=None`` OMITS --panel-frame-from-checkpoint entirely (CHECK 4's path: pre-883
    this was build_panel's unpinned coverage-ranked branch; post-883 it is gate (i)'s refusal)."""
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
    return cmd


def eval_checkpoint(work, frame_path, ckpt_dir, out_dir, tag, position_size_pct=None,
                     weight_rebalance=True):
    cmd = _eval_cmd(frame_path, ckpt_dir, out_dir, tag, position_size_pct, weight_rebalance)
    _run(cmd, "eval_vector.py (--operator policy-argmax)")
    # eval_vector names its output <out_dir>/results_<tag>_<ckpt-label>.json
    matches = [f for f in os.listdir(out_dir) if f.startswith(f"results_{tag}_") and f.endswith(".json")]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one eval_vector artifact for tag={tag!r}, found {matches}")
    with open(os.path.join(out_dir, matches[0])) as fh:
        return json.load(fh)


def eval_checkpoint_expect_refusal(work, frame_path, ckpt_dir, out_dir, tag,
                                    position_size_pct=None, weight_rebalance=True):
    """TASK_883: CHECKS 2/4 now expect a REFUSAL (non-zero exit, no artifact written), not a
    parsed result. Does NOT raise on non-zero exit — returns the CompletedProcess so the
    caller can assert on returncode/stderr and that no `results_<tag>_*` artifact appeared."""
    cmd = _eval_cmd(frame_path, ckpt_dir, out_dir, tag, position_size_pct, weight_rebalance)
    print(f"    $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=_REPO, capture_output=True, text=True, timeout=280)


_EV_MODULE = None


def _eval_vector_module():
    """`eval_vector.py` imported as a module (the 861/863 smoke pattern) so CHECK 2b can call
    `build_panel`/`evaluate_checkpoint` DIRECTLY — the in-process path, which does not run
    `main()`'s TASK_883 universe guard."""
    global _EV_MODULE
    if _EV_MODULE is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("ev_882_inproc", EVAL_PY)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _EV_MODULE = mod
    return _EV_MODULE


def _inproc_args(ev, frame_path, position_size_pct=0.1):
    """The eval_vector CLI defaults CHECK 1's subprocess runs under, as a namespace (mirrors
    `tools/tests/test_decision_rule.py::fixture_smoke`). `--teacher-floor` is OFF here: it adds
    a SEPARATE floor leg and leaves the operator walk (the number CHECK 2b compares) untouched,
    and CHECK 2b re-proves that by reproducing CHECK 1's subprocess net@21 exactly."""
    import types
    return types.SimpleNamespace(
        db=FIXTURE_DB, symbols=None, start=START, end=END, bar_minutes=BAR_MINUTES,
        n_assets=N_ASSETS, max_decisions=None, eval_seed=0,
        position_size_pct=position_size_pct,
        panel_source="minute", panel_frame_from_checkpoint=frame_path,
        panel_frame_source="minute",
        # left UNSET (None) on purpose: `_resolve_require_active` then applies the SAME
        # REQUIRE_ACTIVE_SLOTS_DEFAULT the CLI default gives CHECK 1's subprocess.
        require_active_slots=None,
        operator="policy-argmax", model_size="small", h_cycles=4, l_cycles=6, sims=32,
        discount=0.997, gumbel_scale=3.0, value_scale=1.0, value_shift=0.0,
        vector_k=16, vector_hold_bias=3.0, vector_dither_scale=1.0,
        vector_proposal_scheme="dithered-book", vector_proposal_temp=1.0,
        vector_proposal_explore_slots=2, vector_per_node_resample=False,
        vector_budget_step=False, vector_budget_pct=0.99,
        vector_weight_rebalance=True, vector_weight_unit=-1.0, vector_weight_max=4.0,
        vector_rebalance_deadband=-1.0,
        prior_weight=1.0, leaf_eval="model", leaf_rollout_r=3, leaf_vhat_tail=True,
        decision_rule="argmax", decision_tau=0.5, decision_m=4,
        max_positions=16, min_cash_threshold=100.0, teacher_floor=False, tag="inproc882",
    )


def inproc_net21(frame_path, ckpt_dir, position_size_pct=0.1):
    """net@21 for `ckpt_dir` graded on the panel built from `frame_path`, IN-PROCESS.

    This is CHECK 2b's instrument. It exists because TASK_883 made the subprocess path refuse a
    pin that disagrees with the checkpoint's own provenance — correct, and exactly what CHECK 2
    now pins — but that refusal fires BEFORE the model ever sees the mismatched frame, so the
    subprocess can no longer demonstrate the premise both 882 and 883 rest on: that the encoder
    is POSITIONALLY sensitive (slot k IS asset k), i.e. that a permuted frame really does change
    the number. `main()` owns the guard; `build_panel`/`evaluate_checkpoint` do not, so calling
    them directly measures the mechanism without weakening the guard."""
    ev = _eval_vector_module()
    args = _inproc_args(ev, frame_path, position_size_pct)
    panel = ev.build_panel(args)
    label = ev._ckpt_label(ckpt_dir)
    model, trainer = ev.build_model_and_trainer(args, args.n_assets, init_seed=0)
    ev.load_checkpoint_into(trainer, ckpt_dir)
    result = ev.evaluate_checkpoint(label, model, trainer.state.params, None, panel, args,
                                    ev.load_corridor_constants(), ev.load_ext_windows())
    return result["tiers"]["21"]["sharpe_full"]


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

        # ---- CHECK 2 (TASK_883 gate ii): frame-mismatch now REFUSES --------------------
        print("[882/883] CHECK 2 -- eval_vector.py under the MISMATCHED (reversed) frame "
              "(must REFUSE, TASK_883 gate ii) ...")
        r2 = eval_checkpoint_expect_refusal(work, frame_b, ckpt, work, "evalB")
        no_artifact2 = not [f for f in os.listdir(work) if f.startswith("results_evalB_")]
        refused2 = (r2.returncode != 0 and "MISMATCH" in r2.stderr
                    and "first differing slot:" in r2.stderr)
        status2 = "PASS" if (refused2 and no_artifact2) else "FAIL"
        print(f"    [{status2}] CHECK 2 frame-mismatch REFUSAL "
              f"(returncode={r2.returncode}, refused={refused2}, no_artifact={no_artifact2})")
        if status2 == "FAIL":
            failures.append(
                "CHECK 2 (frame-mismatch REFUSAL, TASK_883 gate ii) — a checkpoint graded "
                "under a REVERSED slot order, against a checkpoint that carries its OWN "
                "TASK_874 provenance recording the NATURAL order, did not REFUSE naming the "
                f"mismatch. returncode={r2.returncode}\nstderr tail:\n{r2.stderr[-2000:]}")

        # ---- CHECK 2b: the PREMISE, still measured — positional sensitivity -------------
        # CHECK 2's refusal is the right outcome but it fires before the model sees the
        # mismatched frame, so it can no longer PROVE that a permuted frame changes the
        # number — the premise 882 and 883 both stand on. Measured here on the in-process
        # path (build_panel + evaluate_checkpoint directly; main()'s guard is not involved
        # and is not weakened). Two legs: the matched frame must reproduce CHECK 1's
        # subprocess net@21 (the instrument is the same computation), and the reversed frame
        # must then DIVERGE beyond the retired CHECK 2's tolerance.
        print("[882] CHECK 2b -- IN-PROCESS positional sensitivity (build_panel + "
              "evaluate_checkpoint, no main() guard) ...")
        net21_inproc_a = inproc_net21(frame_a, ckpt)
        net21_inproc_b = inproc_net21(frame_b, ckpt)
        print(f"    in-process MATCHED frame : net@21={net21_inproc_a!r} "
              f"(subprocess CHECK 1: {net21_eval_a!r})")
        print(f"    in-process REVERSED frame: net@21={net21_inproc_b!r}")
        same_as_subproc = rel_or_abs_close(net21_eval_a, net21_inproc_a, rtol=1e-6, atol=1e-6)
        diverges = not rel_or_abs_close(net21_inproc_a, net21_inproc_b, rtol=0.05, atol=0.3)
        status2b = "PASS" if (same_as_subproc and diverges) else "FAIL"
        print(f"    [{status2b}] CHECK 2b positional sensitivity "
              f"(in-process==subprocess on the matched frame={same_as_subproc}, "
              f"permuted frame diverges={diverges})")
        if status2b == "FAIL":
            failures.append(
                "CHECK 2b (positional sensitivity, in-process) — "
                + ("the in-process matched-frame read did NOT reproduce CHECK 1's subprocess "
                   f"net@21 ({net21_inproc_a!r} vs {net21_eval_a!r}), so this leg is not "
                   "measuring the same computation and its divergence claim means nothing. "
                   if not same_as_subproc else "")
                + ("a checkpoint graded under a REVERSED slot order matched its own "
                   f"correctly-framed read within tolerance ({net21_inproc_b!r} vs "
                   f"{net21_inproc_a!r}): the slot-identity mechanism 882 and 883 both rest "
                   "on no longer reproduces on the fixture — re-derive the mechanism before "
                   "trusting CHECK 1's or CHECK 2's pass." if not diverges else ""))

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

        # ---- CHECK 4 (TASK_883 gate i): the pin OMITTED entirely now REFUSES ------------
        print("[882/883] CHECK 4 -- eval_vector.py with NO --panel-frame-from-checkpoint at "
              "all (must REFUSE, TASK_883 gate i) ...")
        r4 = eval_checkpoint_expect_refusal(work, None, ckpt, work, "evalUnpinned")
        no_artifact4 = not [f for f in os.listdir(work) if f.startswith("results_evalUnpinned_")]
        refused4 = (r4.returncode != 0 and "REFUSING an unpinned" in r4.stderr
                    and "--panel-frame-from-checkpoint" in r4.stderr
                    and "--allow-unpinned-universe" in r4.stderr)
        status4 = "PASS" if (refused4 and no_artifact4) else "FAIL"
        print(f"    [{status4}] CHECK 4 unpinned run REFUSES "
              f"(returncode={r4.returncode}, refused={refused4}, no_artifact={no_artifact4})")
        if status4 == "FAIL":
            failures.append(
                "CHECK 4 (pin OMITTED entirely, TASK_883 gate i) — an UNPINNED eval_vector "
                "checkpoint-grading run did not REFUSE naming the missing "
                "--panel-frame-from-checkpoint flag and the --allow-unpinned-universe escape "
                f"hatch, and/or wrote a results artifact anyway. returncode={r4.returncode}\n"
                f"stderr tail:\n{r4.stderr[-2000:]}")

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
    print("[882] PASS — all 4 checks + the in-process premise leg 2b (+ fixture-unchanged) "
          "green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
