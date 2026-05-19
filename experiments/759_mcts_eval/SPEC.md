# 759_mcts_eval — Eval-Time MCTS Sweep (Path A)

**Status**: READY-TO-LAUNCH
**Owner**: server agent
**Implementation**: commit `bcc5be1` (workstation tip post deploy)
**Reference**: `tasks/TASK_759_PLANNING_PIVOT.md`, lore §149

## Goal

Close the documented gap between current benchmark (argmax action selection) and live trading (`real_trade_jax.py:131,326` uses MCTS). The benchmark has been systematically under-reporting the deployed model's behavior since `real_trade_jax.py` was wired to MCTS.

**Test**: re-evaluate existing 741_mint baseline checkpoints with eval-time MCTS at sims ∈ {0 (argmax), 16, 32, 64}, both decodes (`joint`, `per_asset`), and measure Δα vs the argmax baseline.

This is a **pure benchmark sweep on existing checkpoints — no retraining.**

## Why this and not the train-time sweep

Phase-1 review (operator-approved): if we ran train-time `mcts_sims` sweeps with the current argmax benchmark, the result would conflate (a) "deeper search at train" with (b) "we're measuring it with the wrong eval operator." Path A (this SPEC) tests the eval-side question in isolation. Path B (train-side sweep with eval-MCTS enabled) becomes interpretable only after Path A is in hand.

## Mechanism

```bash
python tools/benchmark_easy.py \
  --checkpoint <ckpt_path> \
  --periods-file config/benchmark_periods.json \
  --eval-mcts-sims <N> \
  --eval-mcts-decode <joint|per_asset> \
  --output <out_path>
```

- `--eval-mcts-sims 0` (or omitted) → argmax path (unchanged behavior, verified by smoke).
- `--eval-mcts-sims N>0` → MCTS via `MuZeroMCTXSearch` with `discount=0.95` (matches training `mcts_discount`), `gumbel_scale=0.0` (deterministic), PopArt scale/shift loaded from ckpt metadata, per-window rng = `PRNGKey(window_idx)`.
- `--eval-mcts-decode joint` → take mctx's single joint action, default other assets to Hold (matches `real_trade_jax.py`).
- `--eval-mcts-decode per_asset` → decompose `policy_probs` to `(B, n_assets, 5)` and argmax per asset.

## Variants (8 cells = 4 sims × 2 decodes minus duplicates)

| sims | decode | what it tests |
|---:|---|---|
| 0 | — | argmax baseline (existing `benchmark.json` reference) |
| 16 | joint | minimum search, deployment-truthful |
| 16 | per_asset | minimum search, factored policy decode |
| 32 | joint | matches training-time sims, deployment-truthful |
| 32 | per_asset | matches training-time sims, factored |
| 64 | joint | deeper search, deployment-truthful |
| 64 | per_asset | deeper search, factored |

## Seeds & checkpoints

Three checkpoints to sweep (the 741_mint baselines used everywhere as pairing reference):

- `runs/741_mint/baseline_seed_7/phase2_stageB_best.pkl`
- `runs/741_mint/baseline_seed_13/phase2_stageB_best.pkl`
- `runs/741_mint/baseline_seed_23/phase2_stageB_best.pkl`

If the `phase2_stageB_best.pkl` files are not at those paths on server, locate them (server agent finds the correct path via `find runs/741_mint -name 'phase2_stageB_best.pkl'`) and adjust.

## Output layout

```
runs/759_mcts_eval/
├── seed_7/
│   ├── argmax/benchmark.json
│   ├── sims16_joint/benchmark.json
│   ├── sims16_per_asset/benchmark.json
│   ├── sims32_joint/benchmark.json
│   ├── sims32_per_asset/benchmark.json
│   ├── sims64_joint/benchmark.json
│   └── sims64_per_asset/benchmark.json
├── seed_13/...
├── seed_23/...
└── result.md
```

## Wall estimate

Per Phase-1 review:
- argmax baseline: ~3 min per period × ~10 periods × 3 seeds = ~90 min (probably already exists in `runs/741_mint/eval_baseline_seed_*/benchmark.json`; reuse).
- sims=16: ~10-15× slower → ~15-22 hours total across 3 seeds.
- sims=32: ~20-30× slower → ~30-45 hours total.
- sims=64: ~40-60× slower → ~60-90 hours total.

**Recommended execution**: per-seed, per-sim. Don't parallelize sims across same GPU (each eval run already saturates the model). 2-wide across seeds OK.

**Triage**: launch sims=16 (joint + per_asset) for all 3 seeds first (~20h total, 2-wide ~10h wall). If results are interesting (any direction), proceed to sims=32 and sims=64. If sims=16 shows no movement vs argmax, consider stopping.

## Pre-launch gates

1. **Workstation smoke already passed** (commit `bcc5be1` report):
   - Argmax regression: diff 3.34e-05 vs reference — argmax path unchanged.
   - MCTS @ sims=16 joint: alpha +0.11204, 53 trades, plausible, bit-identical determinism across two runs.
2. **Server-side smoke before sweep**: re-run argmax baseline on one 741_mint checkpoint, confirm it matches the existing `benchmark.json` summary to within 1e-3 (catches config drift between workstation and server). Then run one sims=16 joint on the same checkpoint and confirm no crash.

## Promotion gate

This is a **diagnostic sweep, not a mechanism-promotion test.** Outcomes:

- **If eval-MCTS lifts Δα significantly (any decode, any sims)**: the benchmark/live gap is real and material. All historical campaign numbers are systematically biased (argmax-based). Plan:
  1. Adopt eval-MCTS as the new benchmark default for future campaigns.
  2. Re-run a few key historical evaluations under eval-MCTS to recalibrate the campaign scorecard.
  3. Then Path B (train-side sims sweep with eval-MCTS enabled) becomes the next probe.
- **If eval-MCTS leaves Δα flat or worse**: argmax has been a fine proxy for live (within noise). The benchmark/live gap is small, no recalibration needed. Move on to TASK_759 P2 (value-target MC mixture).

## Caveats (from Phase-1 review, all addressed in implementation)

- **C1**: both decodes implemented per operator decision.
- **C2**: `discount=0.95` passed to `MuZeroMCTXSearch`, not the 0.997 default.
- **C3**: PopArt `value_scale=sqrt(popart.var+eps)`, `value_shift=popart.mean` loaded from ckpt metadata. RuntimeError if missing.
- **C4**: `gumbel_scale=0.0` and `rng = PRNGKey(window_idx)` for per-window determinism.
- **C5**: 34ch sequential path's per-step state machinery preserved; MCTS sits inside it.
- **C6**: all-ones action mask, shape `(B, n_assets*5)`.

## What is NOT in this SPEC

- No training. No new checkpoints.
- No baseline pairing changes — the argmax `benchmark.json` is the baseline reference.
- No live-trader changes. `real_trade_jax.py` already does MCTS, untouched.

## Server execution checklist

1. Pull workstation training branch (tip ≥ `bcc5be1`).
2. Verify `tools/benchmark_easy.py` has `--eval-mcts-sims` flag (smoke).
3. Confirm 741_mint baseline .pkl checkpoints exist.
4. Server-side smoke (one ckpt, one period, sims=16 joint).
5. Launch sweep: sims=16 (joint+per_asset) × 3 seeds, 2-wide.
6. As results arrive: tag flags `/tmp/eval_mcts_s{N}_{decode}_seed_{S}_done.flag`.
7. After sims=16 closes: operator (workstation) decides whether to continue sims=32 or stop.

## Auto-continuation policy (explicit)

**Smoke PASSES (both checks below) → auto-launch the full sims=16 sweep without operator confirmation.**

Smoke pass criteria:
1. Argmax re-run on one 741_mint checkpoint matches the existing `runs/741_mint/eval_baseline_seed_*/benchmark.json` `summary.avg_alpha` to within ≤ 1e-3 (catches workstation↔server config drift).
2. sims=16 joint run on the same checkpoint (one period, e.g. shortest from `config/benchmark_periods.json`) completes without crash. No further numerical gate at smoke — the experiment IS the measurement.

**Smoke FAILS → ABORT the entire experiment, do not launch the sweep.**

Failure modes that must abort:
- Argmax regression > 1e-3 → workstation↔server divergence; do not proceed until reconciled (likely an env / JAX / mctx version mismatch).
- Crash in sims=16 path → implementation bug surfaced on server but not workstation; report and stop.
- Missing PopArt metadata in ckpt → C3 violation; RuntimeError will fire; stop.
- Missing .pkl at expected path → adjust path (per checklist step 3), but if no .pkl is recoverable, abort.

On abort: write `runs/759_mcts_eval/ABORT.md` with the failure reason, log paths, and any stderr excerpt. Set `/tmp/759_smoke_failed.flag` for operator detection. Do NOT silently fall through to argmax-only runs or skip the sweep without leaving the abort artifact.

**After sims=16 sweep closes (3 seeds × 2 decodes done, all flags posted) → STOP and wait for operator.**

The sims=32 and sims=64 tiers are not auto-launched. Operator (workstation) inspects sims=16 result.md, then decides:
- Continue with sims=32 (signal worth amplifying)
- Skip to sims=64 (signal might need more depth)
- Stop the sweep (signal is flat or worse)

Rationale: each subsequent tier doubles compute. Auto-continuation past sims=16 risks burning 30-90h on a flat signal.

## References

- Phase-1 review report: in session transcript (subagent agentId `a293b61b047397b71`).
- Implementation commit: `bcc5be1`.
- Workstation smoke results: argmax diff 3.34e-05, MCTS@16 deterministic.
- `real_trade_jax.py:131, 326` — live trading MCTS path being mirrored here.
- `mcts_trader/jax_agent/mctx_search.py` — wrapped search class.
