# SPEC — 831-EXT: R1 robustness battery (gpu-server)

**Pathway**: R1 extension (`docs/pathway.md`), H1 (search) + a regime read (H2 signal only).
**Pre-registered 2026-07-08, before launch.** Runner = the shipped `experiments/831_search_probe/
search_probe.py` (unchanged); this dir adds only this SPEC + `run_battery.sh`. Budget ≤3h server
(CPU cells + 2 GPU cells), kill date 2026-07-08. **No promotion claims come from this battery** —
it hardens the Stage-3 design inputs (sampler choice, K, value-horizon requirement).

## Questions (pre-registered)
- **A `seeds10`** — full 3×3 grid, seeds 0–9, full window. PASS: dithered-book|K16 stays in the
  top-2 of the net@21 grid means. FAIL ⇒ carried-scheme choice reopens (planner regroom, not a
  silent swap).
- **B `bear22/bull23/y2024/y2025`** — dithered-book, K{8,16,32}, seeds 0–9, per-regime windows.
  Report per-window corridors [floor, ceiling] + search position. Soft check: dithered net@21 ≥
  the per-window in-env floor in ≥3/4 windows. (H2 signal: where in the regime map perfect-model
  search wins/loses — informative for Stage-3, not a verdict.)
- **C `r1/r2/r6/r12`** — dithered|K16, seeds 0–4, value-rollout R ∈ {1,2,6,12} (R=3 = the
  official run). Deliverable: the R-curve, and specifically **min R at which mean net@21 > the
  engine ref +0.751** — this is the value-horizon quality bar the Stage-3 LEARNED value must
  clear. No pass/fail.
- **D `k64/k128`** — dithered-book, K ∈ {64,128}, seeds 0–4, full window, GPU cell. Deliverable:
  does K > 32 keep paying (net@21 AND gross — watch the turnover tax seen at K32)? Informs the
  Stage-3 search budget. No pass/fail.

## Execution
On gpu-server inside tmux (`831x` session): `bash experiments/831_ext/run_battery.sh` —
sequential cells, each under `tools/safe_run.sh` (cgroup 24G; GPU cells pinned to a verified-idle
GPU at MEM_FRACTION=0.7, PREALLOCATE default per feedback rule). Outputs `runs/831_ext/<cell>.json`
+ `.log`, completion marker `runs/831_ext/BATTERY_DONE`. Results pulled to the workstation via
`tools/deploy/pull-results.sh`; verdict lines go to `experiments/LEDGER.md` (one per question).
DB = server `trading_data.db` — **daily-panel parity verified 2026-07-08** (this battery is
daily-only; minute tables there are stale, unused here).
