#!/usr/bin/env python3
"""TASK_803 Phase 1 — token-trade-gate theta-sweep runner.

Orchestrates the canonical scorecard (eval-MCTS sims=16 joint, 3 seed-checkpoints, the canonical
benchmark periods) for a small theta-grid PLUS a theta=OFF baseline, then pairs every (theta, seed,
period) against the SAME checkpoint UNGATED and reports Δα / Δtrade-rate / ΔSharpe. Verdict:
MARGIN-EXISTS / NO-MARGIN.

The scorecard is NOT reimplemented here — we shell out to ``tools/benchmark_easy.py`` (the existing
recal scorecard entry: ``--eval-mcts-sims 16 --eval-mcts-decode joint`` + ``--json``) and read its
per-period JSON. One subprocess per (seed × theta) — the OFF run is just ``--token-gate off``.

CPU-orchestration only: this script spawns GPU eval subprocesses; it does no GPU work itself. The
single ``run_scorecard`` seam is mockable (G3) so the grid iteration / OFF-pairing / RESULT writer
are unit-testable with NO GPU.

Discipline: server-only at real run time (tmux + safe_run, verified-idle GPU, <=2-wide). This file
only BUILDS the commands + aggregates; the main thread launches it on the server.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

# result_writer is pure-stdlib (no JAX/DB) — safe to import on any host.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))   # repo root, for tools.candle_lang.*
from tools.candle_lang import result_writer as RW  # noqa: E402

CANONICAL_PERIODS = ["bull", "bear", "sideways", "ftx_crash", "etf_approval", "boring_week"]
OFF = "OFF"   # sentinel theta label for the ungated baseline


# --------------------------------------------------------------------------- #
# scorecard seam (mockable for G3)
# --------------------------------------------------------------------------- #
@dataclass
class ScoreRequest:
    checkpoint: str
    seed: int
    theta: object              # float, or the OFF sentinel string
    periods: Sequence[str]
    gpu: int
    sims: int
    language: str
    k: int
    mega_bars: int
    codelen_floor: float
    tokenizer: str
    db: str
    benchmark_script: str
    out_json: str


def build_benchmark_cmd(req: ScoreRequest) -> List[str]:
    """Construct the exact ``benchmark_easy.py`` CLI for one (seed, theta) scorecard cell."""
    cmd = [
        sys.executable, req.benchmark_script,
        "--checkpoint", req.checkpoint,
        "--db", req.db,
        "--periods", *req.periods,
        "--eval-mcts-sims", str(req.sims),
        "--eval-mcts-decode", "joint",
        "--json", req.out_json,
    ]
    if req.theta == OFF:
        cmd += ["--token-gate", "off"]
    else:
        cmd += [
            "--token-gate", "hold",
            "--token-gate-theta", str(float(req.theta)),
            "--token-gate-language", req.language,
            "--token-gate-k", str(req.k),
            "--token-gate-mega-bars", str(req.mega_bars),
            "--token-gate-codelen-floor", str(req.codelen_floor),
        ]
        if req.tokenizer:
            cmd += ["--token-gate-tokenizer", req.tokenizer]
    return cmd


def run_scorecard(req: ScoreRequest) -> Dict[str, dict]:
    """Run ONE scorecard cell (subprocess) and return {period_name -> per-period metric dict}.

    This is the ONLY seam that touches the GPU. Unit tests monkeypatch it (G3). The GPU index is
    pinned via CUDA_VISIBLE_DEVICES so the co-tenant contract (feedback_server_vram_envelope) holds.
    """
    cmd = build_benchmark_cmd(req)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(req.gpu)
    print("  $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, env=env, check=True)
    with open(req.out_json) as f:
        payload = json.load(f)
    out: Dict[str, dict] = {}
    for p in payload.get("periods", []):
        out[p["name"]] = p
    return out


# --------------------------------------------------------------------------- #
# sweep orchestration
# --------------------------------------------------------------------------- #
def _metric(period_row: dict, key: str) -> float:
    v = period_row.get(key)
    return float(v) if v is not None else float("nan")


def run_sweep(
    *,
    checkpoints: Dict[int, str],          # {seed -> checkpoint path}
    theta_grid: Sequence[float],
    periods: Sequence[str],
    gpu: int,
    sims: int,
    language: str,
    k: int,
    mega_bars: int,
    codelen_floor: float,
    tokenizer: str,
    db: str,
    benchmark_script: str,
    out_dir: str,
    score_fn: Callable[[ScoreRequest], Dict[str, dict]] = run_scorecard,
) -> dict:
    """Drive the full grid. Returns a results dict (rows + verdict) and writes RESULT.{md,json}.

    For each seed: run the OFF baseline once, then each theta; pair every (theta, period) against the
    SAME seed's OFF cell -> Δα / Δtrade-rate / ΔSharpe.
    """
    os.makedirs(out_dir, exist_ok=True)
    thetas: List[object] = [OFF] + [float(t) for t in theta_grid]

    # raw[seed][theta_label][period] = per-period metric dict
    raw: Dict[int, Dict[object, Dict[str, dict]]] = {}
    for seed, ckpt in checkpoints.items():
        raw[seed] = {}
        for theta in thetas:
            label = "off" if theta == OFF else f"theta{theta}"
            out_json = os.path.join(out_dir, f"seed{seed}_{label}.json")
            req = ScoreRequest(
                checkpoint=ckpt, seed=seed, theta=theta, periods=periods, gpu=gpu, sims=sims,
                language=language, k=k, mega_bars=mega_bars, codelen_floor=codelen_floor,
                tokenizer=tokenizer, db=db, benchmark_script=benchmark_script, out_json=out_json,
            )
            raw[seed][theta] = score_fn(req)

    # pair every gated theta vs OFF, per (seed, period)
    rows: List[dict] = []
    for seed in checkpoints:
        off_cell = raw[seed].get(OFF, {})
        for theta in theta_grid:
            cell = raw[seed].get(float(theta), {})
            for period in periods:
                base = off_cell.get(period)
                gated = cell.get(period)
                if base is None or gated is None:
                    continue
                rows.append({
                    "seed": seed,
                    "theta": float(theta),
                    "period": period,
                    "alpha_off": _metric(base, "model_alpha"),
                    "alpha_gated": _metric(gated, "model_alpha"),
                    "d_alpha_pp": (_metric(gated, "model_alpha") - _metric(base, "model_alpha")) * 100.0,
                    "trade_rate_off": _metric(base, "model_trade_rate"),
                    "trade_rate_gated": _metric(gated, "model_trade_rate"),
                    "d_trade_rate_pp": (_metric(gated, "model_trade_rate")
                                        - _metric(base, "model_trade_rate")) * 100.0,
                    "sharpe_off": _metric(base, "model_sharpe"),
                    "sharpe_gated": _metric(gated, "model_sharpe"),
                    "d_sharpe": _metric(gated, "model_sharpe") - _metric(base, "model_sharpe"),
                })

    summary = _summarize(rows, theta_grid)
    verdict = summary["verdict"]
    _write_result(out_dir, rows, summary, theta_grid, periods, checkpoints,
                   language=language, k=k, mega_bars=mega_bars, codelen_floor=codelen_floor, sims=sims)
    return {"rows": rows, "summary": summary, "verdict": verdict}


def _summarize(rows: List[dict], theta_grid: Sequence[float]) -> dict:
    """Per-theta paired Δα stats + the MARGIN-EXISTS / NO-MARGIN verdict.

    Verdict (mirrors the TASK_759 light bar): MARGIN-EXISTS if ANY theta has paired mean Δα ≥ +1pp
    AND no seed-mean Δα < −3pp; else NO-MARGIN. (Per-seed mean = mean over periods for that seed.)
    """
    per_theta = {}
    best_mean = float("-inf")
    margin = False
    for theta in theta_grid:
        tr = [r for r in rows if r["theta"] == float(theta)]
        if not tr:
            continue
        d_alpha = np.array([r["d_alpha_pp"] for r in tr], dtype=float)
        d_trade = np.array([r["d_trade_rate_pp"] for r in tr], dtype=float)
        d_sharpe = np.array([r["d_sharpe"] for r in tr], dtype=float)
        seeds = sorted({r["seed"] for r in tr})
        seed_means = []
        for s in seeds:
            sa = np.array([r["d_alpha_pp"] for r in tr if r["seed"] == s], dtype=float)
            seed_means.append(float(np.nanmean(sa)))
        mean_d_alpha = float(np.nanmean(d_alpha))
        worst_seed_mean = float(np.nanmin(seed_means)) if seed_means else float("nan")
        per_theta[float(theta)] = {
            "mean_d_alpha_pp": mean_d_alpha,
            "min_d_alpha_pp": float(np.nanmin(d_alpha)),
            "max_d_alpha_pp": float(np.nanmax(d_alpha)),
            "mean_d_trade_rate_pp": float(np.nanmean(d_trade)),
            "mean_d_sharpe": float(np.nanmean(d_sharpe)),
            "seed_mean_d_alpha_pp": seed_means,
            "worst_seed_mean_d_alpha_pp": worst_seed_mean,
        }
        best_mean = max(best_mean, mean_d_alpha)
        if mean_d_alpha >= 1.0 and worst_seed_mean >= -3.0:
            margin = True
    return {
        "per_theta": per_theta,
        "best_mean_d_alpha_pp": best_mean if per_theta else float("nan"),
        "verdict": "MARGIN-EXISTS" if margin else "NO-MARGIN",
    }


def _write_result(out_dir, rows, summary, theta_grid, periods, checkpoints,
                  *, language, k, mega_bars, codelen_floor, sims) -> str:
    """Render runs/803_gate/RESULT.{md,json} via the shared result_writer."""
    md_path = os.path.join(out_dir, "RESULT.md")

    hdr = ("| theta | mean Δα(pp) | min/max Δα(pp) | mean Δtrade%(pp) | mean ΔSharpe | "
           "worst seed-mean Δα(pp) |")
    sep = "|---|---|---|---|---|---|"
    table = [hdr, sep]
    for theta in theta_grid:
        st = summary["per_theta"].get(float(theta))
        if st is None:
            continue
        table.append(
            f"| {theta:g} | {st['mean_d_alpha_pp']:+.2f} | "
            f"{st['min_d_alpha_pp']:+.2f}/{st['max_d_alpha_pp']:+.2f} | "
            f"{st['mean_d_trade_rate_pp']:+.2f} | {st['mean_d_sharpe']:+.3f} | "
            f"{st['worst_seed_mean_d_alpha_pp']:+.2f} |"
        )

    per_cell = ["| seed | theta | period | Δα(pp) | Δtrade%(pp) | ΔSharpe |",
                "|---|---|---|---|---|---|"]
    for r in rows:
        per_cell.append(
            f"| {r['seed']} | {r['theta']:g} | {r['period']} | {r['d_alpha_pp']:+.2f} | "
            f"{r['d_trade_rate_pp']:+.2f} | {r['d_sharpe']:+.3f} |"
        )

    sections = [
        ("Setup", [
            f"- eval-MCTS sims={sims}, decode=joint, paired vs SAME-checkpoint UNGATED (theta=OFF).",
            f"- seeds (checkpoints): {dict(checkpoints)}",
            f"- periods: {list(periods)}",
            f"- gate: language={language}, k={k}, mega_bars={mega_bars}, codelen_floor={codelen_floor}",
            f"- theta-grid: {list(theta_grid)}",
        ]),
        ("Per-theta paired summary (Δ vs ungated)", table),
        ("Per (seed × theta × period) pairs", per_cell),
        ("Read", [
            f"- best mean Δα across the grid: {summary['best_mean_d_alpha_pp']:+.2f}pp.",
            "- MARGIN-EXISTS if any theta has paired mean Δα ≥ +1pp AND no seed-mean Δα < −3pp.",
        ]),
    ]
    meta = {
        "verdict": summary["verdict"],
        "best_mean_d_alpha_pp": summary["best_mean_d_alpha_pp"],
        "sims": sims, "language": language, "k": k, "mega_bars": mega_bars,
        "codelen_floor": codelen_floor, "theta_grid": list(theta_grid),
        "seeds": list(checkpoints.keys()), "periods": list(periods),
        "per_theta": summary["per_theta"],
    }
    return RW.write_result(md_path, "TASK_803 Phase 1 — token trade gate θ-sweep",
                           sections, summary["verdict"], meta)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_checkpoints(seed_args: List[str]) -> Dict[int, str]:
    """Parse repeated --checkpoint SEED=PATH entries into {seed -> path}."""
    out: Dict[int, str] = {}
    for s in seed_args:
        if "=" not in s:
            raise ValueError(f"--checkpoint must be SEED=PATH, got {s!r}")
        seed, path = s.split("=", 1)
        out[int(seed)] = path
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="TASK_803 Phase 1 token-gate θ-sweep runner")
    ap.add_argument("--checkpoint", action="append", required=True, metavar="SEED=PATH",
                    help="Repeatable. One per seed, e.g. --checkpoint 7=runs/760_.../seed_7/best.pkl")
    ap.add_argument("--theta-grid", type=float, nargs="+", required=True,
                    help="θ values (trailing-k mean codelength thresholds, bits). OFF baseline auto-added.")
    ap.add_argument("--periods", nargs="+", default=CANONICAL_PERIODS,
                    help=f"Default: the canonical scorecard periods {CANONICAL_PERIODS}.")
    ap.add_argument("--gpu", type=int, default=3, help="GPU index (pinned via CUDA_VISIBLE_DEVICES).")
    ap.add_argument("--sims", type=int, default=16, help="eval-MCTS sims (canonical scorecard = 16).")
    ap.add_argument("--language", default="L1", help="Token language short code (L1/L8). Default L1.")
    ap.add_argument("--k", type=int, default=8, help="Trailing window (bars) for the codelength mean.")
    ap.add_argument("--mega-bars", type=int, default=16, help="Mega/flat run-length threshold (bars).")
    ap.add_argument("--codelen-floor", type=float, default=0.0, help="Bits; >0 enables codelen-floor arm.")
    ap.add_argument("--tokenizer", default="", help="Frozen L1 vocab JSON path (required for L1).")
    ap.add_argument("--db", default="trading_data.db", help="Trading DB path.")
    ap.add_argument("--benchmark-script", default=str(_HERE.parent / "benchmark_easy.py"),
                    help="Path to benchmark_easy.py.")
    ap.add_argument("--out-dir", default="runs/803_gate", help="Output dir for RESULT.{md,json}.")
    args = ap.parse_args(argv)

    checkpoints = _parse_checkpoints(args.checkpoint)
    t0 = time.time()
    res = run_sweep(
        checkpoints=checkpoints, theta_grid=args.theta_grid, periods=args.periods, gpu=args.gpu,
        sims=args.sims, language=args.language, k=args.k, mega_bars=args.mega_bars,
        codelen_floor=args.codelen_floor, tokenizer=args.tokenizer, db=args.db,
        benchmark_script=args.benchmark_script, out_dir=args.out_dir,
    )
    print(f"\nVERDICT: {res['verdict']}  (best mean Δα "
          f"{res['summary']['best_mean_d_alpha_pp']:+.2f}pp)  in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
