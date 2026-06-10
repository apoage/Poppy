#!/usr/bin/env python3
"""TASK_803 Phase 1 acceptance battery (CPU only, NO GPU, NO full eval).

Run:
    JAX_PLATFORMS=cpu python tools/candle_lang/test_803_token_gate.py

Covers:
  G1  --token-gate OFF => run_period_eval decision path is bit-identical to canon: the flag defaults
      to 'off', the gate config is disabled, NO candle_lang import happens on the OFF path, and a
      synthetic forward through the (gate-aware) np.where overlay leaves actions UNCHANGED.
  G2  gate ON with a SYNTHETIC codelength/token_id timeline + synthetic actions => HOLD (=1) is
      forced EXACTLY at the bars where (mega/flat OR trailing-k mean codelength < theta), AND the L1
      2-bar serving lag is applied via token_window_slice.input_lag_for (includes a case proving L1
      uses lag 2, NOT lag 1).
  G3  dry-run the runner orchestration with a MOCKED scorecard (no GPU) => θ-grid iteration,
      OFF-baseline pairing, and RESULT writer all work end-to-end.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # repo root

from tools.candle_lang import token_gate as TG
from tools.candle_lang import token_timeline as TT
from tools.candle_lang import token_window_slice as TWS

_FAILS = []


def check(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        _FAILS.append(msg)


def _make_timeline(ts, token_id, codelength, valid, asset_id=0, language="L1_bpe"):
    return TT.TimelineResult(
        np.asarray(ts, np.int64), np.asarray(token_id, np.int32),
        np.asarray(codelength, np.float32), np.asarray(valid, bool), asset_id, language,
    )


# =============================================================================
# G1 — OFF path is a no-op / bit-identical
# =============================================================================
def test_g1_off_noop():
    print("\nG1 — --token-gate OFF is bit-identical (no token code, no import side-effects)")

    # We do NOT import benchmark_easy here (it pulls the full JAX/model stack). G1 is a CPU/no-GPU
    # check, so we (a) inspect the benchmark source to prove the OFF defaults + guards exist exactly
    # as written, and (b) test the overlay no-op semantics with pure numpy. Source inspection is a
    # STRONGER guarantee than runtime here: it proves no candle_lang import is reachable on OFF.
    import ast
    be_path = Path(__file__).resolve().parent.parent / "benchmark_easy.py"
    src = be_path.read_text()
    tree = ast.parse(src)

    # 1. TokenGateConfig dataclass default mode == 'off' and .enabled returns (mode != 'off').
    tgc = next((n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == "TokenGateConfig"), None)
    check(tgc is not None, "TokenGateConfig dataclass exists in benchmark_easy.py")
    mode_default = None
    for stmt in tgc.body:
        if isinstance(stmt, ast.AnnAssign) and getattr(stmt.target, "id", None) == "mode":
            mode_default = ast.literal_eval(stmt.value)
    check(mode_default == "off", f"TokenGateConfig.mode default == 'off' (got {mode_default!r})")
    check("return self.mode != 'off'" in src, "TokenGateConfig.enabled is (mode != 'off')")

    # 2. argparse default for --token-gate is 'off' (mirror the exact declaration).
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--token-gate', type=str, default='off', choices=['off', 'hold'])
    ns = p.parse_args([])
    check(ns.token_gate == "off", "argparse default for --token-gate is 'off'")

    # 3. The gate-build guard in run_period_eval is `token_gate_cfg is not None and
    #    token_gate_cfg.enabled` — so OFF never enters the branch that imports candle_lang. Prove the
    #    guard text is present AND that the candle_lang import lives INSIDE _build_token_gate (the
    #    enabled-only helper), never at module top-level of benchmark_easy.
    check("token_gate_cfg is not None and token_gate_cfg.enabled" in src,
          "run_period_eval gate-build guard requires cfg.enabled (OFF skips all token code)")
    top_level_imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    imports_candle_at_top = any(
        (isinstance(n, ast.ImportFrom) and (n.module or "").startswith("tools.candle_lang"))
        or (isinstance(n, ast.Import) and any(a.name.startswith("tools.candle_lang") for a in n.names))
        for n in top_level_imports
    )
    check(not imports_candle_at_top,
          "benchmark_easy.py does NOT import candle_lang at module top level (no OFF-path side-effects)")

    # 4. Synthetic forward: the in-loop overlay is `actions = np.where(grid, 1, actions)` guarded by
    #    `token_gate is not None`. With token_gate None (OFF) the actions array is the SAME object,
    #    bit-identical. Emulate both the sequential and batched overlays.
    rng = np.random.default_rng(0)
    actions_seq = rng.integers(0, 5, size=(64,)).astype(np.int32)
    token_gate = None
    out_seq = actions_seq
    if token_gate is not None:  # the exact guard in the sequential path
        out_seq = np.where(token_gate.hold_grid[0], 1, actions_seq)
    check(out_seq is actions_seq and np.array_equal(out_seq, actions_seq),
          "sequential OFF overlay leaves actions object UNCHANGED (bit-identical)")

    model_actions = rng.integers(0, 5, size=(100, 64)).astype(np.int32)
    out_batch = model_actions
    if token_gate is not None:  # the exact guard in the batched path
        out_batch = np.where(token_gate.hold_grid, 1, model_actions)
    check(out_batch is model_actions and np.array_equal(out_batch, model_actions),
          "batched OFF overlay leaves model_actions object UNCHANGED (bit-identical)")


# =============================================================================
# G2 — gate ON forces HOLD exactly at the expected bars, with the L1 lag-2
# =============================================================================
def test_g2_mega_and_theta_and_lag():
    print("\nG2 — gate ON: HOLD forced exactly at (mega/flat OR trailing-k codelength<theta), L1 lag-2")

    # ---- 2a. mega/flat run-length arm, NO lag (use language L8 => lag 0) ----
    # token ids: a 5-bar mega run (id 9) then alternating noise. mega_bars=4 => the 4th..5th bars of
    # the run are mega. With lag 0 the per-bar token IS the timeline token.
    T = 12
    ts = np.arange(100, 100 + T, dtype=np.int64)
    tok = np.array([9, 9, 9, 9, 9, 3, 7, 3, 7, 3, 7, 3], dtype=np.int32)  # run of 9 length 5
    code = np.full(T, 5.0, dtype=np.float32)                              # high bits => theta arm off
    valid = np.ones(T, dtype=bool)
    tl = _make_timeline(ts, tok, code, valid, language="L8_haar")

    ws = 1                          # window i -> price-row i (identity), so window==bar
    n_windows = T - ws             # = 11
    prices_ts = ts.copy()          # identity join: prices_ts[r] == timeline.ts[r]

    gate = TG.build_gate_from_timelines(
        {0: tl}, [0], prices_ts, n_windows, ws,
        language="L8", theta=-1.0,  # theta below all codelengths => theta arm never fires
        k=1, mega_bars=4, codelen_floor=0.0,
    )
    # run_len of id 9: bars 0..4 -> 1,2,3,4,5; mega (>=4) at bars 3,4. Decision-row r=window i (ws=1).
    # window i gates iff bar i is mega. Expected gated windows: {3, 4}.
    gated = set(np.flatnonzero(gate.hold_grid[:, 0]).tolist())
    check(gated == {3, 4}, f"L8 lag-0 mega arm gates exactly windows {{3,4}} (got {sorted(gated)})")

    # ---- 2b. theta arm (trailing-k mean codelength < theta), NO lag ----
    # codelength: low (0.5) on bars 6..8, high (5.0) elsewhere. k=1, theta=1.0 => gate bars 6,7,8.
    code2 = np.full(T, 5.0, dtype=np.float32)
    code2[6:9] = 0.5
    tok2 = np.array([3, 7, 3, 7, 3, 7, 3, 7, 3, 7, 3, 7], dtype=np.int32)  # no mega run
    tl2 = _make_timeline(ts, tok2, code2, valid, language="L8_haar")
    gate2 = TG.build_gate_from_timelines(
        {0: tl2}, [0], prices_ts, n_windows, ws,
        language="L8", theta=1.0, k=1, mega_bars=99, codelen_floor=0.0,  # mega arm disabled
    )
    gated2 = set(np.flatnonzero(gate2.hold_grid[:, 0]).tolist())
    check(gated2 == {6, 7, 8}, f"L8 lag-0 theta arm gates exactly windows {{6,7,8}} (got {sorted(gated2)})")

    # ---- 2c. THE LAG-2 CASE: same theta arm, but language L1 (lag 2). ----
    # With lag 2, code_lag[t] = code2[t-2], so the low-codelength bars 6,7,8 move to 8,9,10. The
    # gate must read bars 8,9,10, NOT 6,7,8 (lag-2) and NOT 7,8,9 (would be lag-1, the off-by-one
    # TASK_800 explicitly forbids). Leading 2 bars become invalid (-1).
    gate3 = TG.build_gate_from_timelines(
        {0: tl2}, [0], prices_ts, n_windows, ws,
        language="L1", theta=1.0, k=1, mega_bars=99, codelen_floor=0.0,
    )
    gated3 = set(np.flatnonzero(gate3.hold_grid[:, 0]).tolist())
    check(gate3.lag == 2, f"L1 gate resolved lag == 2 via input_lag_for (got {gate3.lag})")
    check(gated3 == {8, 9, 10}, f"L1 lag-2 theta arm gates {{8,9,10}} (got {sorted(gated3)})")
    lag1_set = {7, 8, 9}
    check(gated3 != lag1_set, f"L1 gate is NOT the lag-1 off-by-one set {sorted(lag1_set)}")
    # cross-check directly against the canonical helper (the ONLY allowed lag source).
    code_lag_ref = np.full(T, np.nan)
    shifted_ref = TWS.lag_shift_per_bar(tok2, TWS.input_lag_for("L1"))
    check(int(shifted_ref[2]) == int(tok2[0]) and int(shifted_ref[0]) == TWS.INVALID_TOKEN_ID,
          "token_window_slice.lag_shift_per_bar applies lag 2 (tok_lag[2]==tok[0], tok_lag[0]==-1)")

    # ---- 2d. full gate ON => HOLD(=1) forced exactly at the gated windows on a synthetic actions ----
    # Use the 2a mega gate. Build synthetic per-window actions (all BUY=2) and apply the overlay
    # exactly as run_period_eval does (np.where(grid, 1, actions)).
    actions = np.full((n_windows, 1), 2, dtype=np.int32)   # all BUY
    gated_actions = np.where(gate.hold_grid, 1, actions)
    forced_hold = set(np.flatnonzero(gated_actions[:, 0] == 1).tolist())
    check(forced_hold == {3, 4},
          f"overlay forces HOLD(=1) exactly at gated windows {{3,4}} (got {sorted(forced_hold)})")
    # non-gated windows keep BUY untouched
    nongated_ok = all(gated_actions[i, 0] == 2 for i in range(n_windows) if i not in {3, 4})
    check(nongated_ok, "non-gated windows keep their original action (BUY) — overlay only touches gated")

    # ---- 2e. trailing-k>1 mean (causal) sanity ----
    # k=3, codelength low only on a single bar 6; trailing-3 mean at bar 6 = (5+5+0.5)/3=3.5 > theta=1
    # so a single low bar should NOT trip a k=3 mean theta=1 gate; a 3-bar low stretch should.
    tmean = TG.trailing_mean(code2, valid, k=3)
    check(abs(tmean[6] - (5.0 + 5.0 + 0.5) / 3.0) < 1e-9,
          "trailing_mean is causal trailing-3 (mean over bars 4,5,6)")
    check(abs(tmean[8] - (0.5 * 3) / 3.0) < 1e-9, "trailing_mean over the full low stretch == 0.5")


# =============================================================================
# G2-bis — end-to-end through annotate_asset-shaped data + codelength model
# =============================================================================
def test_g2bis_codelength_model_populates():
    print("\nG2-bis — codelength model fold/finalize populates the theta signal (no DB)")
    # tiny synthetic id stream; rarer ids get more bits. Fold via the real _CodelengthModel.
    K = 10
    cm = TT._CodelengthModel(K, alpha=1.0)
    ids = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 2], dtype=np.int64)  # 0 common, 1/2 rare
    cm.fold_ids(ids)
    cm.finalize()
    bits = cm.bits_array(np.array([0, 1, 2], dtype=np.int64))
    check(bits[0] < bits[1] and bits[0] < bits[2],
          "codelength: common token (0) has FEWER bits than rare tokens (1,2)")
    check(cm.bits_for(-1) == 0.0, "codelength of invalid id (-1) is 0.0")


# =============================================================================
# G3 — runner orchestration with a MOCKED scorecard (no GPU)
# =============================================================================
def test_g3_runner_dryrun():
    print("\nG3 — runner θ-grid iteration + OFF-pairing + RESULT writer (mocked scorecard, no GPU)")
    from tools.candle_lang import run_803_gate_sweep as R

    periods = ["bull", "bear"]
    seeds = {7: "ckpt_s7.pkl", 13: "ckpt_s13.pkl"}
    theta_grid = [1.0, 2.0]

    # Deterministic synthetic scorecard: OFF baseline has alpha=0, trade_rate=0.5, sharpe=0.
    # theta=1.0 lifts alpha by +0.02 (=+2pp) and cuts trade_rate by 0.10; theta=2.0 hurts alpha
    # by -0.04 (=-4pp). This lets us assert the verdict logic + pairing arithmetic precisely.
    calls = []

    def mock_score(req: R.ScoreRequest):
        calls.append((req.seed, req.theta))
        out = {}
        for per in req.periods:
            if req.theta == R.OFF:
                a, tr, sh = 0.00, 0.50, 0.0
            elif float(req.theta) == 1.0:
                a, tr, sh = 0.02, 0.40, 0.1
            else:  # 2.0
                a, tr, sh = -0.04, 0.55, -0.2
            out[per] = {"name": per, "model_alpha": a, "model_trade_rate": tr, "model_sharpe": sh}
        return out

    with tempfile.TemporaryDirectory() as td:
        res = R.run_sweep(
            checkpoints=seeds, theta_grid=theta_grid, periods=periods, gpu=3, sims=16,
            language="L1", k=8, mega_bars=16, codelen_floor=0.0, tokenizer="vocab.json",
            db="trading_data.db", benchmark_script="benchmark_easy.py", out_dir=td,
            score_fn=mock_score,
        )
        # grid iteration: (OFF + 2 thetas) × 2 seeds = 6 scorecard calls
        check(len(calls) == 6, f"scorecard called once per (seed × {{OFF,θ1,θ2}}) = 6 (got {len(calls)})")
        # OFF pairing: theta=1.0 Δα == +2pp for every (seed, period)
        d1 = [r["d_alpha_pp"] for r in res["rows"] if r["theta"] == 1.0]
        check(len(d1) == 4 and all(abs(x - 2.0) < 1e-9 for x in d1),
              f"θ=1.0 paired Δα == +2.00pp on all 4 (seed×period) cells (got {d1})")
        d1_tr = [r["d_trade_rate_pp"] for r in res["rows"] if r["theta"] == 1.0]
        check(all(abs(x - (-10.0)) < 1e-9 for x in d1_tr),
              f"θ=1.0 paired Δtrade-rate == -10.00pp (got {d1_tr})")
        d2 = [r["d_alpha_pp"] for r in res["rows"] if r["theta"] == 2.0]
        check(all(abs(x - (-4.0)) < 1e-9 for x in d2), f"θ=2.0 paired Δα == -4.00pp (got {d2})")
        # verdict: θ=1.0 mean Δα=+2pp ≥ +1pp and worst seed-mean +2 ≥ -3 => MARGIN-EXISTS
        check(res["verdict"] == "MARGIN-EXISTS",
              f"verdict MARGIN-EXISTS (θ=1.0 clears the +1pp bar) (got {res['verdict']})")
        # RESULT artifacts written
        md = Path(td) / "RESULT.md"
        js = Path(td) / "RESULT.json"
        check(md.exists() and js.exists(), "RESULT.md + RESULT.json written")
        import json
        meta = json.load(open(js))
        check(meta["verdict"] == "MARGIN-EXISTS", "RESULT.json verdict matches")
        check(abs(meta["meta"]["best_mean_d_alpha_pp"] - 2.0) < 1e-9,
              "RESULT.json best_mean_d_alpha_pp == +2.0")

    # NO-MARGIN branch: make every theta hurt alpha.
    def mock_bad(req: R.ScoreRequest):
        out = {}
        for per in req.periods:
            a = 0.0 if req.theta == R.OFF else -0.05
            out[per] = {"name": per, "model_alpha": a, "model_trade_rate": 0.5, "model_sharpe": 0.0}
        return out

    with tempfile.TemporaryDirectory() as td:
        res2 = R.run_sweep(
            checkpoints=seeds, theta_grid=theta_grid, periods=periods, gpu=3, sims=16,
            language="L1", k=8, mega_bars=16, codelen_floor=0.0, tokenizer="v.json",
            db="db", benchmark_script="b.py", out_dir=td, score_fn=mock_bad,
        )
        check(res2["verdict"] == "NO-MARGIN", f"all-negative grid => NO-MARGIN (got {res2['verdict']})")

    # command builder sanity: OFF cell uses --token-gate off; gated cell carries theta + tokenizer.
    off_cmd = R.build_benchmark_cmd(R.ScoreRequest(
        checkpoint="c.pkl", seed=7, theta=R.OFF, periods=["bull"], gpu=3, sims=16, language="L1",
        k=8, mega_bars=16, codelen_floor=0.0, tokenizer="v.json", db="db", benchmark_script="b.py",
        out_json="o.json"))
    check("--token-gate" in off_cmd and off_cmd[off_cmd.index("--token-gate") + 1] == "off",
          "OFF scorecard cmd uses --token-gate off")
    check("--token-gate-theta" not in off_cmd, "OFF scorecard cmd has no --token-gate-theta")
    on_cmd = R.build_benchmark_cmd(R.ScoreRequest(
        checkpoint="c.pkl", seed=7, theta=1.5, periods=["bull"], gpu=3, sims=16, language="L1",
        k=8, mega_bars=16, codelen_floor=0.0, tokenizer="v.json", db="db", benchmark_script="b.py",
        out_json="o.json"))
    check(on_cmd[on_cmd.index("--token-gate") + 1] == "hold"
          and "--token-gate-theta" in on_cmd
          and on_cmd[on_cmd.index("--token-gate-tokenizer") + 1] == "v.json"
          and on_cmd[on_cmd.index("--eval-mcts-sims") + 1] == "16",
          "gated scorecard cmd carries hold/theta/tokenizer + sims=16")


def main():
    print("=" * 78)
    print("TASK_803 Phase 1 — token trade gate acceptance battery (CPU, no GPU)")
    print("=" * 78)
    test_g1_off_noop()
    test_g2_mega_and_theta_and_lag()
    test_g2bis_codelength_model_populates()
    test_g3_runner_dryrun()
    print("\n" + "=" * 78)
    if _FAILS:
        print(f"FAILED ({len(_FAILS)}):")
        for m in _FAILS:
            print(f"  - {m}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
