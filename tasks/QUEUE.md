# QUEUE — the ranked dark-factory work order

The factory's single entry point (planner-maintained). STATUS banners on task files remain the
truth for what's done; this file only orders what's OPEN. The factory takes from the top, skips
gated rows (noting why), and treats FILLER rows as parallel/anytime slices. Last groomed:
**2026-07-08 post factory run 1** (828 S1+S2 shipped `task828-stage1`/`task828-stage2`; 830/831
factory halves shipped; D6 decided — dual floor).

## Ready — take from the top

| # | Task | Size | Note |
|---|------|------|------|
| 1 | `tasks/TASK_833_RUN_QUEUE_REPAIR.md` | S | fix `tools/deploy/run_queue.sh` (untracked; first commit in-task): subshell BLOCKER at ~:162 + 4 MAJORs — full findings list + gates persisted in the task file (the finding session's patches were NOT persisted; re-derive). Mock-based empirical re-verify required. Redeploy to srv is planner's follow-up. Risk: medium. |
| 2 | `tasks/TASK_829_RETIRE_FOXGLOVE.md` | S–M | decouple `train_phase2.py:484` + `benchmark_easy.py:41` imports, then delete the Foxglove files. PROMOTED ahead of Stage 3: both touch train_phase2.py's import surface, so retire Foxglove FIRST on a clean base. Risk: medium. |
| 3 | `tasks/TASK_828_VECTOR_ACTION_MUZERO.md` (Stage 3) | L | training loop + factored policy targets + ACTION-CONDITIONAL reward head (D2; E3 variance-attribution probe = hard gate). Honor the Stage-2 panel carry-notes in the 828 banner (−inf pre-search masking, 320-wide asset-major head, visit_probs argmax for deterministic eval). Self-play proposal scheme = **dithered-book\|K16** (E2 carried; K8 budget fallback). Needs 829 shipped first. **TRAINING GATE OPEN** (real 831 E1 PASS, 2026-07-08); training itself is still a scheduled run, not factory-loop. Risk: high. |

## Scheduled runs — owner: planner/main session (NOT factory-loop jobs)

**Server status (2026-07-08)**: gpu-server cleared for **DAILY-panel jobs only** — `ohlcv_daily`
+ `assets` synced to workstation parity (table-level; verified 499,430 rows, quick_check ok,
831 cell ran end-to-end in 18s); **minute-bar tables are still Jan-2026 vintage** — full
`sync-db.sh` required before any minute-bar job. Server code current (vector-action spine
deployed); the 7 drifted server files are STASHED there (`git stash list`), not discarded.
Stage-3 training destination = gpu-server GPUs 1/2/3 AFTER the run_queue.sh repair ships.

| Run | What | Where | Status |
|-----|------|-------|--------|
| 830 real corridor | full-panel floors+ceiling+expressibility gap, `trading_data.db` read-only | workstation CPU (screen+safe_run) | **DONE 2026-07-08** — corridor official [+0.424, +29.698] net@21, ref +0.751; H6 armed; LEDGER 830/E0-REAL |
| 831 real E1/E2 sweep | 9 scheme×K combos, full panel, seeds 0,1,2 | workstation CPU (screen+safe_run, ~7 min) | **DONE 2026-07-08** — E1 PASS 9/9 ⇒ Stage-3 training gate OPEN; carried scheme dithered-book\|K16; LEDGER 831/E1-REAL + E2-REAL |

## Gated — do NOT start (named gate must open first)

| Task | Gate |
|------|------|
| `tasks/TASK_832_LINEAR_IMPRINT_EXIT.md` (E5 linear imprint) | 828 Stage 3 banner (831 E1 PASS ✓ satisfied 2026-07-08) |
| `tasks/TASK_828_VECTOR_ACTION_MUZERO.md` (Stage 4 eval) | Stages 1–3 shipped + rung verdicts green; SCHEDULED server run, not a factory-loop job |

## Parked (D1/D4, 2026-07-07 — non-binding "interesting outview"; not factory work)

- `tasks/TASK_824_SERVER_HONEST_REMEASURE.md`, `tasks/TASK_825_CROSSASSET_TREND_CTA.md`,
  `tasks/TASK_826_ONCHAIN_SIGNAL_PROBE.md` (D4: OHLC-only, live within the 24GB DB),
  `tasks/TASK_823_VPS_MICROLIVE_CTA.md` (D0: rejected).

## Grooming rules
Planner re-ranks after each factory run + banners; a row leaves this file only by shipping
(banner) or being explicitly parked. If the factory finds the top row blocked in practice, it
takes the next and records why in its wrap-up.
