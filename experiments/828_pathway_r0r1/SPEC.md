> **STATUS (2026-07-08): HISTORICAL — never executed.** The R1 grid this manifest schedules was
> run OFFICIALLY on the workstation CPU the same morning (LEDGER 831/E1-REAL + E2-REAL; E1 PASS,
> carried scheme dithered-book|K16). Do NOT re-run — double-recording hazard. The manifest+queue
> pattern here is the template for the Stage-3 training sweep (after TASK_833 repairs run_queue.sh).

# SPEC — TASK_828 pathway R0/R1 real-panel sweep (gpu-server)

**What**: the SCHEDULED real-panel runs the dark-factory built the runners for but must not
run in-loop — the E0 corridor (TASK_830) and the E1/E2 search-capacity probe (TASK_831), on the
24GB `trading_data.db` (read-only source of record). Fixture smokes are already green + committed.

**Why now**: TASK_831's **real E1 verdict gates the 828 Stage-3 training launch** (pathway R1),
and its E2 scheme×K ranking picks the Stage-3 sampler. TASK_830's real corridor `[floor, ceiling]`
is the scoreboard every later result is reported against.

## Runner (self-advancing queue)

`tools/deploy/run_queue.sh experiments/828_pathway_r0r1/jobs.txt --sweep 828_pathway_r0r1`

- Schedules the manifest across GPU pool **{1,2,3}** (GPU 0 is VirtualBox-reserved), ≤1 job/GPU,
  each assigned only after a LIVE idle check (never blind `--gpu 1,2,3`).
- Every job runs under `tools/safe_run.sh` (systemd --user cgroup RAM cap + GPU pin +
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.7`). GPU jobs get `JAX_PLATFORMS=cuda` injected (the runners
  default `setdefault(JAX_PLATFORMS, cpu)`, so cuda must be forced).
- CPU jobs (830) don't consume a GPU slot; `--max-cpu` bounds CPU concurrency.
- Per-job logs → `runs/828_pathway_r0r1/<job>.log`; summary → `runs/828_pathway_r0r1/queue_summary.md`.
- Launched inside a tmux session (`queue_828`) so it survives disconnect.

## Jobs

**R0 / TASK_830 corridor is already DONE** (operator ran it on the workstation, safe_run CPU;
corridor OFFICIAL `[+0.424, +29.698]` net@21, ref +0.751; D6 DECIDED dual-floor, H6 armed
conditional — LEDGER `830/E0-REAL`). This sweep runs ONLY the R1 grid:

| job | dev | what | out |
|-----|-----|------|-----|
| `831_<scheme>_K<k>` ×9 | gpu | true-model sampled search, full panel (`--max-decisions -1`), seeds 0,1,2; scheme∈{independent,linear-prior,dithered-book} × K∈{8,16,32} | `runs/828_pathway_r0r1/831_<scheme>_K<k>.json` |

Daily-panel note (from 830/E0-REAL): fixture daily == db daily is confirmed, so these cells produce
the official full-panel verdict whether pointed at `trading_data.db` (used here, source of record) or
the fixture. The gap vs the committed 831 fixture smoke is only the removed decision cap + the 3rd seed.

Device note: 831 cells are set to `gpu` per the operator directive (use up to 3 GPUs). The compute
is many small `step_vector` calls, which can be CPU-favorable — a one-cell timing smoke (GPU vs CPU)
runs first; if CPU wins decisively the manifest device column flips to `cpu` with `--max-cpu 3`.

## Pre-registered verdicts (from the task SPECs — not re-litigated here)

- **E0**: floor reproduces the +0.97 engine bar within tolerance (report the hold-shares in-env
  floor; constant-dollar-weight floor is a documented follow-up per the cleared D6). Report the
  corridor `[floor, ceiling]` per window/cost tier.
- **E1**: any scheme×K reaches the corridor floor (net@21, mean over ≥3 seeds) ⇒ PASS ⇒ Stage-3
  search space contains the book (necessary-not-sufficient). FAIL across ALL schemes ⇒ proposal
  mechanism is the defect ⇒ Stage-3 does NOT launch; planner respecs the sampler.
- **E2**: cheapest scheme×K that reaches the floor → the Stage-3 sampler hint.

## Results handoff

Runner writes `runs/828_pathway_r0r1/`. Operator pulls via `tools/deploy/pull-results.sh`, records
the E0/E1/E2 verdicts + corridor positions in `experiments/LEDGER.md`, and (on E1 PASS) opens the
Stage-3 training row. The server agent does NOT commit results back (SERVER.md contract).

## Constraints

Read-only on `trading_data.db`. Never GPU 0. `MEM_FRACTION=0.7`. tmux-resident. No live orders.
Deterministic (no `--seed` promotion claims off <3 seeds).
