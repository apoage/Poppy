# SPEC — 828 Stage-3 training battery (gpu-server, scheduled run)

**Pathway**: R3 (`docs/pathway.md`). The first LEARNED-model test of H1 — everything until now
was perfect-model. Pre-registered 2026-07-08 before launch. Owner: planner/main session.
Mechanism: TASK_833-repaired `tools/deploy/run_queue.sh`, GPUs {1,2,3}, 1 job/GPU, tmux.

## Jobs (manifest `experiments/828_stage3_train/jobs.manifest`)
Two passes × 3 seeds, all with the 831-EXT-carried operator — dithered-book, K=16,
**hold-bias 3.0 explicit** (CLI default 0.0 is a trap), dither 1.0; daily bars via
`--bar-minutes 1440` on the pinned 821 window/universe (64 assets, never reorder);
`--reward-hybrid --reward-attributable`; small model, sims 32; `--snapshot-every 100`
(immutable mid/late checkpoints — required by the E3/DEC-1 probe, TASK_834).
- **Primary: fee16/slip5 ×3 seeds** (`--assert-postcost`) — the bar condition. The known
  failure mode (821 HOLD collapse) must be beaten UNDER COSTS or not at all.
- **Diagnostic: cost0 ×3 seeds** (no assert-postcost) — existence proof / cost-attribution
  if the fee pass degenerates.

## Pre-flight gates (in order, before the battery)
1. **Deploy** current master (Stage-3 code) to srv; verify tag reachable.
2. **Minute-vintage check** (the training path reads the 1-min `ohlcv` table — recon-verified
   `env_factory.py:289-379` → `vectorbt_env.py:582+`): server `MAX(timestamp) FROM ohlcv` ≥
   **1764543600** (2025-12-01, window end) + spot-check coverage for the pinned 64 symbols.
   FAIL ⇒ full `sync-db.sh` before launch (table-level daily parity does NOT cover this).
3. **GPU smoke**: fee16 template, `--epochs 40 --no-checkpoint`, one verified-idle GPU.
   PASS = env ~1731 steps × 64 assets, finite losses, pct_hold trending <99, and a steady-state
   s/epoch measurement. **Pre-registered budget rule**: if s/epoch × 2000 > 20h/seed, cut
   `--epochs` to fit ≤20h/seed (snapshots every 100 keep partials usable) — recorded, not tuned.

## Monitoring & verdicts
- In-loop generalization eval is fail-closed SKIPPED in vector mode (panel finding) — monitor
  `logs/training_<name>.csv`: reward/alpha, per-asset action histograms, pct_hold,
  `mcts_visit_entropy` (vector-correct in CSV; DS24 shows cosmetic 0.0 — known), losses.
- **HOLD-collapse tripwire (the 821 signature)**: pct_hold pinned >99 with entropy floor-crash
  past epoch ~300 on ≥2 fee seeds ⇒ record FAIL early, don't ride it to 2000.
- **E3/DEC-1 gate (mid/late snapshots)**: `R_model ≥ 0.5 × R_true` via the TASK_834 runner
  (fallback absolute ≥0.5 if R_true leg late). This decides whether the search plans over a
  reward model that feels its own actions — the D2 question, now empirical.
- Stage-4 full-timeline eval vs the corridor/ref stays a SEPARATE gated run (its faithful
  per-asset eval harness is also not built yet — future factory row).
- Artifacts: server `jax_ckpt_fast/<name>/` + `logs/training_<name>.csv` + `runs/828_stage3/`
  → pulled via `pull-results.sh`; verdict lines → `experiments/LEDGER.md` (per pass, per seed).
