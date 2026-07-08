# Handoff 2026-07-08 (factory run 2) — queue scheduler repaired, Foxglove retired

Second dark-factory implementer run of the day (after the vector-action spine run + the server-ops
interlude). Two tasks shipped, both panel-verified, committed and tagged. Both are direct Stage-3
enablers: the unattended multi-GPU scheduler that will drive Stage-3 training is now correct, and
`train_phase2.py`'s import surface is clean for Stage-3 to thread `--vector-action` through.

## Shipped this run

| Commit | Tag | Task | What landed | Verification |
|--------|-----|------|-------------|--------------|
| `3cec698` | `task833-run-queue-repair` | TASK_833 | `tools/deploy/run_queue.sh` (first-tracked) — the 5 DO-NOT-SHIP findings fixed + a committable mock harness | Medium panel + independent re-verify on **real systemd scopes**; 16 harness scenarios; selftest 7/7 |
| `40afa14` | `task829-retire-foxglove` | TASK_829 | retire Foxglove: 1 import retarget + 15 file deletions | Medium panel; zero behavior change proven; selftest 7/7 |

## TASK_833 — run_queue.sh (the fail-closed multi-GPU scheduler)

The scheduler built during the server-ops interlude had failed its verification workflow
(DO-NOT-SHIP). This task fixed it:
- **BLOCKER**: `acquire_idle_gpu` ran in a `$(...)` subshell → its `FREE_GPUS` mutation was lost →
  every GPU job piled onto GPU 1 and OOM'd. Now sets a parent-scope `ACQUIRED_GPU`.
- **+4 MAJOR**: GPU reservation at acquire (no TOCTOU); cleanup `trap` so `systemd --user` scope
  units never orphan; malformed manifest lines fail loud; `--acquire-timeout` + heartbeat (no
  infinite spin).
- **Panel found 3 defects IN the fixes, all fixed + re-verified**: N1 (MEDIUM) — the per-`--sweep`
  flock overclaimed cross-instance mutual exclusion; two queues with different sweeps could
  double-book. Moved the lock to a **host-global `runs/.locks/gpu<N>.lock`** (proven: serializes
  with the shared lock, double-books without it). N2/N3 (LOW) — a GPU acquire-timeout no longer
  abandons CPU jobs; the CPU-slot wait is now bounded + fail-loud.
- Harness (`tools/tests/test_run_queue_scheduler.sh`) drives the REAL script + REAL safe_run
  systemd scopes with a stubbed nvidia-smi; 16 scenarios incl. no-double-book distribution and
  interrupt/cleanup (SIGTERM → trap → 0 orphans → exit 130).

**Planner/main-session follow-up (NOT factory):** the BROKEN copy is still deployed on gpu-server
(DO-NOT-INVOKE). Redeploy the fixed script via `tools/deploy/deploy.sh` before any unattended
queue run. (This is the same one-way deploy used earlier today; the training branch is current.)

## TASK_829 — Foxglove retired

- Retargeted the one live coupling `train_phase2.py:484` `from train_easy import SegmentPool` →
  `from mcts_trader.data.segment_pool import SegmentPool`. Proven no-op: the deleted `train_easy.py`
  was a pure re-export shim of that identical class (extracted TASK_772.5).
- Deleted 15 files (`train_easy.py`, `easy_trainer.py`, `train_easy_loop.py`, `run_dual_easy.sh`,
  `bridge_trainer.py`, `merge_checkpoints.py`, `tools/orchestration/{e-dojo_v1,dual_dojo,reset_edojo,
  reset_dual_dojo}.sh/.py`, `tools/DSe/`). Kept `easy_config.py` (the canonical `benchmark_easy`
  scorecard needs it) and the non-Foxglove `tools/orchestration/` lineage (dojo_v2/v3/v4, train_manager, …).
- Deletion + import-retarget only; zero behavior change; no LIVE missed couplings
  (dynamic/importlib/subprocess/shell sweep clean); `benchmark_easy --help` RC=0; selftest 7/7.

## Where it stands — Stage 3 is next and gate-open

The vector-action pathway: Stages 1–2 shipped, R0/R1 instruments run (830 corridor official
`[+0.424, +29.698]` net@21; **831 real E1 = PASS ⇒ Stage-3 TRAINING GATE OPEN**, carried sampler
**dithered-book|K16**, K8 budget fallback). Both blockers to Stage 3 are now cleared:
- the unattended GPU scheduler is correct (TASK_833),
- `train_phase2.py`'s import surface is clean (TASK_829).

**TASK_828 Stage 3** (queue row 3, L, risk high) is the next build: thread `--vector-action`
through `train_jax_fast.py`/config/`env_factory`/self-play; factored-visit policy targets; the
**action-conditional reward head (D2)** with the **E3 variance-attribution probe as a hard gate**.
Honor the Stage-2 panel carry-notes (in the 828 banner): −inf pre-search masking, 320-wide
asset-major head, `visit_probs` argmax for deterministic eval. It was left for a fresh session
because (a) it's the largest, highest-risk piece and THE architecture bet, and (b) its training +
E3 gate are a SCHEDULED server run (not a factory-loop job) — the CODE can land ahead, but the
validation that matters needs the GPU run.

## Server state (recap, still true)

gpu-server is cleared for DAILY-panel jobs — `ohlcv_daily` synced to workstation parity
(table-level, 499,430 rows, quick_check ok, 831 cell ran in 18s). **Minute-bar tables are still
Jan-2026 vintage** — a full `sync-db.sh` is required before any minute-bar job. Server code is
current (vector-action spine deployed, commit `c4a27e0`); the 7 drifted server files are STASHED
there (`git stash list`), not discarded. Stage-3 training destination = gpu-server GPUs 1/2/3 AFTER
run_queue.sh is redeployed.

## Note for the next session (process anomaly)

During this run's verify panel, a subagent's tool-result output contained an embedded
"Verify @tasks/TASK_824…" instruction (inside a `<result>` block, not a user turn). It was treated
as untrusted content and NOT acted on. If TASK_824 verification is actually wanted, it should come
as a direct request. Flagging in case it recurs (possible prompt-injection surface in subagent
outputs).
