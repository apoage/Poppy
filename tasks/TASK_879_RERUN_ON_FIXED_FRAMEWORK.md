# TASK_879 — Re-run the closed lanes on the FIXED framework (operator directive, 2026-08-05)

**Status**: PLANNER-SCHEDULED program (not a factory row — these are GPU batteries the planner
launches; the factory's part is only that the framework fixes land first).
**Size: L (days of server compute, cheap in tokens). Risk: low** (benchmark/training reads, D17:
no gate status).

## The directive (operator, 2026-08-05)

> "if there is experiment on known broken framework we shall rerun experiment on fixed
> framework .. because experiment might get fixed"

This is the most consequential standing instruction in the ledger's history, because **almost
every negative verdict on file was measured on a framework we have since fixed.** A FAIL measured
inside a broken environment does not falsify the hypothesis — it falsifies the run.

## Framework epochs (what changed, and when)

| epoch | fix | landed |
|---|---|---|
| E-a | `step_vector` snapshot-cash accounting → weight-rebalance env (the "cash wall" that made breadth lethal) | TASK_840/843/845/846 |
| E-b | zero-price tail → forward-fill + disclosure (D30) | `task865…` 2026-07-31 |
| E-c | ingest identity split → alias union; interior frozen-cell handling (D31 a1/a2) | `task868…` 2026-08-03 |
| E-d | UTC window resolution; masked daily floor (D31 a3) | `task869…` 2026-08-03 |
| E-e | XBT↔BTC fold — one more slot of real data (TASK_877) | pending |
| E-f | no-print = FAILED FILL, intent preserved in BC targets (TASK_878) | pending |

Everything ledgered before E-b/E-c/E-d ran on a panel that (i) zeroed delisted tails, (ii) could
not see ~6 assets' worth of split-identity data, and (iii) resolved windows in local time. Every
pre-E-a training run additionally learned inside a bankruptcy-inducing cash bug.

## Re-run ladder (ranked by "the framework is the likeliest cause of the FAIL")

1. **828s3 phase-1 self-play battery (cost0)** — the strongest candidate. LEDGER `828s3/env-wall`
   asks literally "why did every phase-1 instrument fail/degenerate?" and answers: the
   `step_vector` cash wall. That defect is FIXED (E-a) and the runs were never repeated. This is
   also the "base model training" block the operator wants the idle GPUs on. 3 seeds, long.
2. **ExIt continuation (E5-exit / E5-exit2)** — failed twice, and D26 parked the lane; but both
   attempts predate E-b/E-c/E-d, and E5-exit2's specific failure (visit-distribution CE rounding
   to 100% HOLD) interacts directly with the mask semantics TASK_878 changes. Re-arm ONLY after
   878 lands — otherwise it re-runs against the same target-rewriting hazard. Un-parks QUEUE row D
   (the ExIt frame pin) the moment this is scheduled.
3. **Search-leaf distillation (857 v̂-corridor / 860 co-distill)** — best −4.23 / −4.02 vs floor
   +0.424, "residual structural in the dynamics embedding". Both pre-E-c. Cheaper than 1 and 2;
   worth one clean re-measure before the lane stays parked.
4. **The instrument reads** (856 rule-grid, 854 inversion, 851 isolation ladder) — cheap,
   CPU/short-GPU, and they calibrate everything above. Do these first as a warm-up: they either
   reproduce (framework-independent) or move (which is itself a finding).

## Ledger discipline change (apply immediately, planner-owned)

Add a **framework-epoch stamp** to the ledger header and treat every pre-E-c verdict as
**PROVISIONAL**: citable as history, NOT citable as "closed" in any argument that gates work.
When a re-run lands, the new row supersedes the old and says so explicitly. This is the same
correction class as the 2026-07-06 "scoped to the old architecture" reframing in MEMORY.md —
honest evidence about a formulation, not a law of nature.

## Sequencing

E-e (877) and E-f (878) land first — re-running before the framework is actually fixed just
manufactures a third set of numbers. The walk-forward battery now running is already on
E-a..E-d and is unaffected. Then ladder rung 4 (cheap calibration), then rung 1 (the long base
training the compute is for), then 3, then 2.

## Out of scope

The pre-vector scalar-architecture negatives (734–810) — superseded by construction, not by
framework: the action space itself changed, so re-running them would measure a different
experiment. They stay closed with their existing scope note.
