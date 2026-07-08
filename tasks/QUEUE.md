# QUEUE — the ranked dark-factory work order

The factory's single entry point (planner-maintained). STATUS banners on task files remain the
truth for what's done; this file only orders what's OPEN. The factory takes from the top, skips
gated rows (noting why), and treats FILLER rows as parallel/anytime slices. Last groomed:
**2026-07-08 post factory run 1** (828 S1+S2 shipped `task828-stage1`/`task828-stage2`; 830/831
factory halves shipped; D6 decided — dual floor).

## Ready — take from the top

| # | Task | Size | Note |
|---|------|------|------|
| 1 | `tasks/TASK_834_E3_ATTRIBUTION_RUNNER.md` | S | checkpoint-loading E3/DEC-1 runner (R_model vs R_true) — the RUNNING Stage-3 battery needs it when mid/late snapshots exist (~hours). New module `experiments/834_e3_probe/`, zero canon diffs. Risk: low-medium. |
| 2 | `tasks/TASK_832_LINEAR_IMPRINT_EXIT.md` (E5 linear imprint) | M | SCOPE CONFIRMED (planner 2026-07-08): pure CODE build — `experiments/832_linear_imprint/imprint.py` runner + SPEC + fixture smoke; **buffer-injection** of on-the-fly teacher trajectories (one-hot marginals through `JAXReplayBuffer(vector_action=True).add_trajectory` — the shipped factored CE IS the BC loss). **Reject scope creep into the monsters**: `amax_imprint`/`expert_bc` are scalar-only by design, do NOT extend them; no trainer/env edits. Pin in SPEC: (a) reduced entropy_coef/min_entropy for the imprint phase (the adaptive entropy push fights one-hot targets); (b) **panel consistency** — teacher trajectories walk the TRAINING env panel (`create_jax_env_historical`, 1-min→daily resample), with book signs computed on that same panel, NOT the corridor's daily-close panel (obs-distribution mismatch evaporates the warm start); (c) ExIt half = `--load-checkpoint --reset-optimizer --clear-buffer` (already shipped). ε and verdicts = DEC-2, scheduled run. Risk: medium. |

_Shipped 2026-07-08 factory run 3: **TASK_828 Stage 3** (`task828-stage3` — vector-action training
loop + D2 action-conditional reward head + factored-marginal targets; parity MATCH==PINNED, E3 probe
green, full high-tier panel 0 BLOCKER/MAJOR + 1 MINOR fixed). Stage-3 TRAINING itself is the scheduled
gpu-server GPU run below, not a factory-loop job._
_Shipped 2026-07-08 factory run 2: TASK_833 (`task833-run-queue-repair` — repaired scheduler
redeployed to srv same day) and TASK_829 (Foxglove retired, `40afa14`)._

## Scheduled runs — owner: planner/main session (NOT factory-loop jobs)

**Server status (2026-07-08)**: gpu-server cleared for **DAILY-panel jobs only** — `ohlcv_daily`
+ `assets` synced to workstation parity (table-level; verified 499,430 rows, quick_check ok,
831 cell ran end-to-end in 18s); **minute-bar tables are still Jan-2026 vintage** — full
`sync-db.sh` required before any minute-bar job. Server code current (vector-action spine
deployed); the 7 drifted server files are STASHED there (`git stash list`), not discarded.
Stage-3 training destination = gpu-server GPUs 1/2/3 — run_queue.sh REPAIRED (TASK_833) and
redeployed 2026-07-08; the scheduler is cleared for use.

| Run | What | Where | Status |
|-----|------|-------|--------|
| 830 real corridor | full-panel floors+ceiling+expressibility gap, `trading_data.db` read-only | workstation CPU (screen+safe_run) | **DONE 2026-07-08** — corridor official [+0.424, +29.698] net@21, ref +0.751; H6 armed; LEDGER 830/E0-REAL |
| 831 real E1/E2 sweep | 9 scheme×K combos, full panel, seeds 0,1,2 | workstation CPU (screen+safe_run, ~7 min) | **DONE 2026-07-08** — E1 PASS 9/9 ⇒ Stage-3 training gate OPEN; carried scheme dithered-book\|K16; LEDGER 831/E1-REAL + E2-REAL |
| 828 Stage-3 training | fee16 ×3 seeds (primary) + cost0 ×3 (diagnostic), dithered-book\|K16 **hold-bias 3.0**, snapshots every 100; SPEC+manifest `experiments/828_stage3_train/` | gpu-server GPUs 1/2/3 via run_queue.sh (tmux `q_828s3`) | **LAUNCHING 2026-07-08** — pre-flight: deploy → minute-vintage check (`MAX(ts) ohlcv ≥ 1764543600` — the training path READS THE 1-MIN TABLE, recon-verified) → 40-epoch GPU smoke → battery. E3 verdict via TASK_834 runner on snapshots. |

## Gated — do NOT start (named gate must open first)

| Task | Gate |
|------|------|
| `tasks/TASK_828_VECTOR_ACTION_MUZERO.md` (Stage 4 eval) | Stages 1–3 shipped ✓ (2026-07-08) + Stage-3 training run's rung verdicts green; SCHEDULED server run, not a factory-loop job |

_TASK_832 (E5 linear imprint) moved to Ready 2026-07-08 — its gate (828 Stage 3 banner) is now satisfied._

## Parked (D1/D4, 2026-07-07 — non-binding "interesting outview"; not factory work)

- `tasks/TASK_824_SERVER_HONEST_REMEASURE.md`, `tasks/TASK_825_CROSSASSET_TREND_CTA.md`,
  `tasks/TASK_826_ONCHAIN_SIGNAL_PROBE.md` (D4: OHLC-only, live within the 24GB DB),
  `tasks/TASK_823_VPS_MICROLIVE_CTA.md` (D0: rejected).

## Grooming rules
Planner re-ranks after each factory run + banners; a row leaves this file only by shipping
(banner) or being explicitly parked. If the factory finds the top row blocked in practice, it
takes the next and records why in its wrap-up.
