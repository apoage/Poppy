> **STATUS (2026-07-08): SHIPPED (tag `task833-run-queue-repair`).** All 5 findings fixed +
> first-tracked, with a committable mock harness (16 scenarios, real systemd scopes). Verify panel
> found 3 defects IN the fixes — N1 (MEDIUM: per-sweep flock overclaimed cross-instance safety →
> host-global `runs/.locks/`), N2/N3 (LOW: CPU jobs skipped on GPU timeout; unbounded CPU wait) —
> all fixed + independently re-verified (two-sweep no-double-book proven; N2/N3 bounded/fail-loud;
> selftest 7/7). **Remainder (planner/main session, NOT factory):** redeploy to gpu-server via
> `deploy.sh` to replace the broken live-but-inert copy; until then the server copy is DO-NOT-INVOKE.

# TASK_833 — Repair tools/deploy/run_queue.sh (multi-GPU job scheduler)

**Status**: READY 2026-07-08 (queue row 1). **Size: S. Risk: medium** — this is the fail-closed
scheduler that will drive UNATTENDED Stage-3 training across gpu-server GPUs 1/2/3; a scheduling
bug there means silent OOM or orphaned jobs at 3am, not a red test.

## Context
`tools/deploy/run_queue.sh` (199 lines, currently UNTRACKED — first commit happens in this task)
was built 2026-07-08 by the server-ops session and deployed to gpu-server, then failed its
verification workflow: **DO-NOT-SHIP**. The findings below were empirically verified in that
session but its patches were NOT persisted — re-derive them; the findings list is the scope.

## Verified findings (the exact scope — fix all five, nothing else)
1. **BLOCKER — subshell GPU acquisition**: `g="$(acquire_idle_gpu)"` (line ~162) runs
   `acquire_idle_gpu` in a command-substitution SUBSHELL, so its `FREE_GPUS` mutation
   (`unset`/compact at lines ~79-84) is discarded in the parent. Result (empirically
   reproduced): every job pins to the first pool GPU (GPU 1) and OOMs while GPUs 2-3 idle.
   Fix shape: make acquisition mutate parent state (e.g. set a global + call without
   substitution, or return via nameref) — any fix must keep the live-idle check.
2. **MAJOR — VRAM TOCTOU / no reservation**: the live-idle check and the job launch are not
   atomic; two dispatches can pass the check before either allocates. Add a reservation
   (mark the GPU taken at acquisition, not at job start).
3. **MAJOR — no cleanup trap**: on script exit/CTRL-C/kill, running `systemd-run --scope`
   children are orphaned. Add a `trap` that reaps or at least reports scope units on exit.
4. **MAJOR — silently-dropped manifest lines**: malformed/unparseable `jobs.txt` lines are
   skipped without a report. Fail loudly (list dropped lines; non-zero exit if any).
5. **MAJOR — unbounded idle-wait**: when no GPU frees up, the wait loop spins forever with no
   timeout/heartbeat. Add a configurable timeout + periodic status line.

## Gates
- `bash -n` clean; shellcheck (if available) no new errors.
- **Empirical re-verify of #1/#2**: a dry-run mode or mock (`nvidia-smi` stub) demonstrating a
  3-job manifest distributes across a 2-GPU pool (not all to the first GPU) and that a full
  pool queues rather than double-books. This was the DO-NOT-SHIP reproduction — reproduce it
  green. No real GPU work needed; the fixture-first rule applies (mock, not gpu-server).
- Commit `tools/deploy/run_queue.sh` (first tracking) + the test/mock harness.
- `python -m trader selftest` untouched-green (no repo code paths touched).

## After shipping (planner/main session, NOT this task)
Redeploy to gpu-server via `tools/deploy/deploy.sh` so the broken live-but-inert copy is
replaced. Until then the server copy is DO-NOT-INVOKE (recorded in ops memory + QUEUE).

## Out of scope
`tools/safe_run.sh` (works), `tools/deploy/deploy.sh`, the `experiments/828_pathway_r0r1/`
manifest (historical — R1 already executed on the workstation, see its SPEC banner), any
refactor beyond the five findings.
