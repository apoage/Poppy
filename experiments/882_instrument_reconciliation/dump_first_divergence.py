#!/usr/bin/env python3
"""TASK_882 gate (ii) — retained per-decision divergence probe.

The RESOLUTION claims "the first divergent decision is decision 0". That claim previously
had no retained artifact, so a future reader could not re-verify it without redoing the
probe by hand. THIS script IS the probe, and `first_divergence.json` / `.txt` next to it
are its output.

What it does (fixture only, CPU, no GPU, no `trading_data.db`):

  1. Trains ONE tiny checkpoint on `fixtures/trading_fixture.db` pinned to FRAME_A (the
     fixture's natural 8-symbol slot order) — the same hermetic recipe
     `test_instrument_agreement.py` uses, imported from it so there is one source of truth.
  2. Runs `eval_vector.py --operator policy-argmax` on that checkpoint TWICE, IN-PROCESS,
     changing exactly one variable: the `--panel-frame-from-checkpoint` pin.
       leg "matched"    -> FRAME_A   (slot k == the asset the net trained slot k on)
       leg "mismatched" -> FRAME_B   (the SAME 8 symbols, reversed slot order)
  3. Records, per decision k, on BOTH legs:
       - `raw`  : the pre-mask per-asset argmax vector straight out of
                  `eval_vector.policy_decision_actions` (the net's own output);
       - `exec` : the post-mask vector `rollout` actually executed (TASK_853 price-validity
                  HOLD + TASK_868 causal open-gate HOLD applied);
       - the realized book (`pos_side`), `pos_size`, `cash` and PV after the step, walked on
         the net@21 tier with the same `jit_batch_step_vector` / `compute_portfolio_value`
         primitives `rollout` uses.
  4. Reports the FIRST decision index at which the two legs differ, prints both legs' full
     state at that decision side by side, and counts differing decisions under BOTH
     definitions (raw stream and executed stream) — the two counts are NOT the same number
     and quoting one without naming it is how "466/472" and "465/471" both got reported for
     the same phenomenon.

Recording is strictly additive: `rollout` and `policy_decision_actions` are wrapped, their
return values are passed through untouched, and the recording walk in step 3 is a SEPARATE
walk over an independent copy of the initial state, so the scored numbers this script prints
are the harness's own.

Run:  python experiments/882_instrument_reconciliation/dump_first_divergence.py
"""
import argparse
import json
import os
import shutil
import sys
import tempfile

# MUST precede the first `import jax` anywhere in this process — eval_vector.py sets exactly
# these two before it imports jax, and importing jax first would silently pick a different
# backend (GPU) than the harness runs on. That is both an invariant violation (no GPU jobs)
# and a numeric one: an accidental GPU leg read net@21 -4.9678 where the CPU harness reads
# -4.9876, which would have made this probe disagree with test_instrument_agreement.py.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np                                            # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (_REPO, _HERE, os.path.join(_REPO, "experiments", "835_stage4_eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402

import eval_vector as EV                                      # noqa: E402
from test_instrument_agreement import (                       # noqa: E402
    FIXTURE_DB, FRAME_SYMBOLS, START, END, BAR_MINUTES, N_ASSETS,
    _write_frame, train_checkpoint,
)

NET_TIER_FEE_BPS = 16.0            # eval_vector.FEE_BPS — identifies the net@21 rollout


# --------------------------------------------------------------------------------------
# in-process capture
# --------------------------------------------------------------------------------------
_CAP = {}
_ORIG_PDA = EV.policy_decision_actions
_ORIG_ROLLOUT = EV.rollout


def _patched_policy_decision_actions(*a, **kw):
    out = _ORIG_PDA(*a, **kw)
    _CAP.setdefault("raw_actions", np.asarray(out).copy())
    return out


def _record_walk(prices, env_cfg, state0, policy_actions, n_decisions):
    """Independent per-decision walk of the EXECUTED stream (net@21 tier).

    Mirrors `eval_vector.rollout`'s policy-argmax branch: same step function, same PV
    convention (mark at the row the action executed on, hence current_step - 1).
    """
    state = state0
    rows = []
    for k in range(n_decisions):
        av = jnp.asarray(policy_actions[k], dtype=jnp.int32)[None, :]
        state, _, _, done = EV.jit_batch_step_vector(state, av, prices, env_cfg)
        cur = EV.get_current_prices(prices, state.current_step - 1, state.episode_start_idx)
        pv_arr = jax.vmap(EV.compute_portfolio_value)(state, cur)
        jax.block_until_ready(pv_arr)
        rows.append({
            "k": k,
            "exec_action": np.asarray(av)[0].astype(int).tolist(),
            "pos_side": np.asarray(state.pos_side)[0].astype(int).tolist(),
            "pos_size": [float(x) for x in np.asarray(state.pos_size)[0]],
            "cash": float(np.asarray(state.cash)[0]),
            "pv": float(np.asarray(pv_arr)[0]),
        })
        if bool(np.asarray(done)[0]):
            break
    return rows


def _patched_rollout(params, search_obj, prices, env_cfg, state0, n_assets, n_decisions,
                     eval_seed, max_positions, min_cash_threshold, policy_actions=None,
                     leaf_eval="model", collect=None, open_gate=None):
    rep = _ORIG_ROLLOUT(params, search_obj, prices, env_cfg, state0, n_assets, n_decisions,
                        eval_seed, max_positions, min_cash_threshold,
                        policy_actions=policy_actions, leaf_eval=leaf_eval,
                        collect=collect, open_gate=open_gate)
    if policy_actions is not None and float(env_cfg.fee_bps) == NET_TIER_FEE_BPS:
        _CAP["exec_actions"] = np.asarray(policy_actions).copy()
        _CAP["walk"] = _record_walk(prices, env_cfg, state0, policy_actions, n_decisions)
        _CAP["n_decisions"] = int(n_decisions)
        _CAP["final_pv"] = float(rep["final_pv"])
    return rep


def run_leg(frame_path, ckpt, out_dir, tag):
    """One full `eval_vector.py --operator policy-argmax` run, in-process, recorded."""
    _CAP.clear()
    argv = [
        "eval_vector.py",
        "--db", FIXTURE_DB, "--start", START, "--end", END,
        "--bar-minutes", str(BAR_MINUTES), "--n-assets", str(N_ASSETS),
        "--panel-frame-from-checkpoint", frame_path, "--panel-frame-source", "minute",
        "--operator", "policy-argmax", "--checkpoint", ckpt,
        "--tag", tag, "--out-dir", out_dir,
        "--vector-weight-rebalance", "--teacher-floor",
    ]
    old_argv = sys.argv
    EV.policy_decision_actions = _patched_policy_decision_actions
    EV.rollout = _patched_rollout
    try:
        sys.argv = argv
        EV.main()
    finally:
        sys.argv = old_argv
        EV.policy_decision_actions = _ORIG_PDA
        EV.rollout = _ORIG_ROLLOUT
    leg = dict(_CAP)
    matches = [f for f in os.listdir(out_dir)
               if f.startswith(f"results_{tag}_") and f.endswith(".json")]
    if len(matches) != 1:
        raise RuntimeError(f"expected one artifact for tag={tag!r}, found {matches}")
    with open(os.path.join(out_dir, matches[0])) as fh:
        art = json.load(fh)
    leg["net21"] = art["tiers"]["21"]["sharpe_full"]
    leg["artifact_final_pv"] = art["tiers"]["21"]["final_pv"]
    leg["panel_frame_present"] = ("panel_frame" in art)
    leg["col_symbols"] = art.get("panel", {}).get("panel_frame", {}).get("frame_symbols")
    return leg


def _first_diff(a, b):
    n = min(len(a), len(b))
    for k in range(n):
        if not np.array_equal(np.asarray(a[k]), np.asarray(b[k])):
            return k
    return None if len(a) == len(b) else n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=_HERE,
                    help="where first_divergence.{json,txt} land (default: alongside this file)")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    before = (os.path.getsize(FIXTURE_DB), os.path.getmtime(FIXTURE_DB))
    work = tempfile.mkdtemp(prefix="t882_first_divergence_")
    try:
        frame_a = os.path.join(work, "frame_A.json")
        frame_b = os.path.join(work, "frame_B_reversed.json")
        _write_frame(frame_a, FRAME_SYMBOLS, "matched / natural order")
        _write_frame(frame_b, list(reversed(FRAME_SYMBOLS)), "MISMATCHED / reversed permutation")

        print("[882/divergence] training the fixture checkpoint under FRAME_A ...")
        ckpt = os.path.join(work, "ckpt")
        imp = train_checkpoint(work, frame_a, ckpt, os.path.join(work, "imprint_A.json"))
        imprint_net21 = imp["sanity"]["per_tier"]["21"]["policy_inenv"]
        imprint_pv = imp["sanity"]["per_tier"]["21"]["final_pv"]

        print("[882/divergence] leg A -- eval_vector pinned to the MATCHED frame ...")
        legA = run_leg(frame_a, ckpt, work, "divA")
        print("[882/divergence] leg B -- eval_vector pinned to the MISMATCHED (reversed) frame ...")
        legB = run_leg(frame_b, ckpt, work, "divB")

        rawA, rawB = legA["raw_actions"], legB["raw_actions"]
        exA, exB = legA["exec_actions"], legB["exec_actions"]
        n = int(min(len(rawA), len(rawB)))

        raw_diff_idx = [k for k in range(n) if not np.array_equal(rawA[k], rawB[k])]
        exec_diff_idx = [k for k in range(n) if not np.array_equal(exA[k], exB[k])]
        first_raw = raw_diff_idx[0] if raw_diff_idx else None
        first_exec = exec_diff_idx[0] if exec_diff_idx else None
        walk_first = _first_diff([r["pos_side"] for r in legA["walk"]],
                                 [r["pos_side"] for r in legB["walk"]])

        k0 = first_exec if first_exec is not None else (first_raw if first_raw is not None else 0)
        side_by_side = {
            "decision": k0,
            "matched": {
                "raw_action": np.asarray(rawA[k0]).astype(int).tolist(),
                "exec_action": legA["walk"][k0]["exec_action"],
                "pos_side_after": legA["walk"][k0]["pos_side"],
                "pos_size_after": legA["walk"][k0]["pos_size"],
                "cash_after": legA["walk"][k0]["cash"],
                "pv_after": legA["walk"][k0]["pv"],
            },
            "mismatched": {
                "raw_action": np.asarray(rawB[k0]).astype(int).tolist(),
                "exec_action": legB["walk"][k0]["exec_action"],
                "pos_side_after": legB["walk"][k0]["pos_side"],
                "pos_size_after": legB["walk"][k0]["pos_size"],
                "cash_after": legB["walk"][k0]["cash"],
                "pv_after": legB["walk"][k0]["pv"],
            },
        }

        doc = {
            "schema": "882/first-divergence/v1",
            "produced_by": "experiments/882_instrument_reconciliation/dump_first_divergence.py",
            "db": FIXTURE_DB,
            "window": [START, END],
            "bar_minutes": BAR_MINUTES,
            "n_assets": N_ASSETS,
            "frame_A_symbols": list(FRAME_SYMBOLS),
            "frame_B_symbols": list(reversed(FRAME_SYMBOLS)),
            "variable_changed": "--panel-frame-from-checkpoint ONLY (frame_A vs frame_B)",
            "imprint_sanity_net21": imprint_net21,
            "imprint_sanity_final_pv": imprint_pv,
            "eval_matched_net21": legA["net21"],
            "eval_matched_final_pv": legA["artifact_final_pv"],
            "eval_mismatched_net21": legB["net21"],
            "eval_mismatched_final_pv": legB["artifact_final_pv"],
            "n_decisions_compared": n,
            "n_decisions_leg_matched": int(len(rawA)),
            "n_decisions_leg_mismatched": int(len(rawB)),
            "counts": {
                "raw_stream_differing": len(raw_diff_idx),
                "exec_stream_differing": len(exec_diff_idx),
                "denominator": n,
                "definition": (
                    "raw_stream = the pre-mask per-asset argmax straight out of "
                    "policy_decision_actions; exec_stream = the vector rollout actually "
                    "executed, i.e. raw with the TASK_853 price-validity HOLD and the "
                    "TASK_868 causal open-gate HOLD applied. The two counts differ because "
                    "the masks themselves are frame-dependent: reversing the slot order "
                    "moves which (decision, slot) cells are invalid/frozen, so the executed "
                    "streams can differ at a decision where the raw argmax vectors agree. "
                    "Quote the count WITH its definition; they are not interchangeable."),
            },
            "first_divergent_decision_raw": first_raw,
            "first_divergent_decision_exec": first_exec,
            "first_divergent_decision_realized_book": walk_first,
            "side_by_side": side_by_side,
        }
        out_json = os.path.join(args.out_dir, "first_divergence.json")
        with open(out_json, "w") as fh:
            json.dump(doc, fh, indent=2)

        lines = []
        w = lines.append
        w("TASK_882 gate (ii) — first divergent decision, both instruments side by side")
        w("=" * 88)
        w(f"fixture      : {FIXTURE_DB}")
        w(f"window       : {START} .. {END}   bar_minutes={BAR_MINUTES}  n_assets={N_ASSETS}")
        w(f"frame A      : {FRAME_SYMBOLS}")
        w(f"frame B      : {list(reversed(FRAME_SYMBOLS))}   (same set, reversed slot order)")
        w("variable     : --panel-frame-from-checkpoint ONLY; everything else identical")
        w("")
        w(f"imprint.py sanity replay (frame A)      : net@21 {imprint_net21!r}  PV {imprint_pv!r}")
        w(f"eval_vector policy-argmax, frame A      : net@21 {legA['net21']!r}  PV {legA['artifact_final_pv']!r}")
        w(f"eval_vector policy-argmax, frame B      : net@21 {legB['net21']!r}  PV {legB['artifact_final_pv']!r}")
        w("")
        w(f"decisions compared                      : {n}")
        w(f"raw  (pre-mask argmax) differing        : {len(raw_diff_idx)}/{n}")
        w(f"exec (post-mask, executed) differing    : {len(exec_diff_idx)}/{n}")
        w(f"first divergent decision, raw stream    : {first_raw}")
        w(f"first divergent decision, exec stream   : {first_exec}")
        w(f"first divergent realized book (pos_side): {walk_first}")
        w("")
        w(f"--- decision {k0} state, side by side " + "-" * 45)
        for name in ("matched", "mismatched"):
            s = side_by_side[name]
            w(f"  [{name}]")
            w(f"    raw argmax   : {s['raw_action']}")
            w(f"    executed     : {s['exec_action']}")
            w(f"    pos_side     : {s['pos_side_after']}")
            w(f"    pos_size     : {[round(x, 6) for x in s['pos_size_after']]}")
            w(f"    cash         : {s['cash_after']!r}")
            w(f"    pv           : {s['pv_after']!r}")
        w("")
        w("Reading: the observation window at decision 0 is already built from the wrong")
        w("slot-to-asset mapping under frame B, so the divergence is present from the very")
        w("first decision — there is no 'drift' phase. Slot k IS asset k positionally in the")
        w("encoder; permuting the frame feeds every slot's learned weights another asset's")
        w("price history.")
        out_txt = os.path.join(args.out_dir, "first_divergence.txt")
        with open(out_txt, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print("\n".join(lines))
        print(f"\n[882/divergence] wrote {out_json}")
        print(f"[882/divergence] wrote {out_txt}")
    finally:
        if args.keep:
            print(f"[882/divergence] workdir kept: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)

    after = (os.path.getsize(FIXTURE_DB), os.path.getmtime(FIXTURE_DB))
    if before != after:
        print(f"[882/divergence] FIXTURE CHANGED {before} -> {after} — INVARIANT VIOLATION")
        return 1
    print(f"[882/divergence] fixture unchanged: size={before[0]} mtime={before[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
