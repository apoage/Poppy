# QUEUE — the ranked dark-factory work order

The factory's single entry point (planner-maintained). STATUS banners on task files remain the
truth for what's done; this file only orders what's OPEN. The factory takes from the top, skips
gated rows (noting why), and treats FILLER rows as parallel/anytime slices. Last groomed:
**2026-07-08 post factory run 1** (828 S1+S2 shipped `task828-stage1`/`task828-stage2`; 830/831
factory halves shipped; D6 decided — dual floor).

## Ready — take from the top

| # | Task | Size | Note |
|---|------|------|------|
| 1 | `tasks/TASK_832_LINEAR_IMPRINT_EXIT.md` (E5 linear imprint) | ? | Gate SATISFIED 2026-07-08: 828 Stage 3 shipped (`task828-stage3`, 831 E1 PASS ✓). Planner to confirm scope/size before the factory takes it — verify it is a CODE build (imprint teacher wiring) and not a scheduled GPU run before starting. |

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
| 828 Stage-3 training | multi-epoch `--vector-action` train to convergence + the real E3 R-attribution PASS (DEC-1 `R_model ≥ 0.5×R_true`) on a mid/late model; ≥3 seeds; scheme dithered-book\|K16 | gpu-server GPUs 1/2/3 via run_queue.sh (tmux) | **READY TO SCHEDULE 2026-07-08** — CODE shipped `task828-stage3`, gates green. NOTE: minute-bar tables on srv are Jan-2026 vintage; a daily-panel training run is fine, but confirm data vintage vs the run's needs (full `sync-db.sh` before any minute-bar dependence). |

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
