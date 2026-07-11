> **STATUS (2026-07-11): SHIPPED as `task841-checkpoint-wedge`.** `run_queue.sh --stall-timeout`
> stall watchdog (kill+archive `_wedged_ep<N>`+relaunch-once, second wedge fail-loud). **Default
> OFF** (`STALL_TIMEOUT=0`, opt-in) — orchestrator deviation from "default 2700" per invariant #1
> after the panel showed default-armed false-kills healthy CSV-less/`--name=` jobs at ~55min
> (MAJOR, double-CONFIRMED, FIXED); unarmed summary byte-identical to HEAD → **DECISIONS D11**.
> Root cause characterized (orbax post-save GC vs RLock) but STOPPED at out-of-scope trainer
> surgery; `--debug-stall-dump` wired to capture the next real wedge's stack. Smoke 18/18,
> scheduler 16/16. Follow-up: apply a flag-gated trainer save fix once the stack is captured.

# TASK_841 — Post-checkpoint-save wedge: watchdog mitigation + root-cause hunt

**Status**: READY 2026-07-11. **Size: S–M. Risk: low-medium** (run_queue.sh watchdog = pure
orchestration; root-cause may touch `safe_checkpoint.py`/trainer save path — flag/guard only).

## Why
**2 of 6 seed-runs wedged, identical signature**: training freezes IMMEDIATELY after an orbax
"Finished synchronous save" line — process alive, GPU 0%, no further log/CSV writes, forever.
- phase-1 s0: ep1581, right after `checkpoint_79000` save (2026-07-09, 12h lost).
- phase-1b s0: ep162, right after `checkpoint_8000` save (2026-07-10, 37h lost — workstation
  was off; an unattended overnight run loses its whole window to this).
Both times the SAME manifest line re-run solo completed clean through the wedge epoch —
non-deterministic, not data-positional. Cost so far: ~2 GPU-days + two manual interventions.

## Scope
1. **Watchdog (the value; ship first)**: `run_queue.sh --stall-timeout <sec>` (default 2700):
   a GPU job whose training CSV (`logs/training_<job_id>.csv`, job-declarable glob) goes
   unmodified for the timeout while the process lives is declared WEDGED → kill process group,
   archive partial ckpt dir + logs with a `_wedged_ep<N>` suffix (the established convention),
   **relaunch the same manifest line once** (max 1 auto-retry per job, counted in the summary
   as `WEDGED-RETRIED`); second wedge = FAILED row, fail-loud. No CSV (pre-first-write) uses
   process start + grace period.
2. **Root-cause hunt (bounded)**: reproduce-or-characterize — thread-dump the wedged state
   (`py-spy dump` on a live wedge if one exists, else faulthandler timeout dump wired behind
   `--debug-stall-dump`); prime suspects: orbax synchronous save's cleanup thread vs the
   training loop's RLock (JAX pitfalls note), tqdm/logging interaction. If root cause is found
   and the fix is a config/guard (e.g. orbax `enable_async_checkpointing=False` already?
   check), ship it flag-gated; if it needs invasive trainer surgery, STOP and report — the
   watchdog already caps the blast radius.
3. Fixture smoke for the watchdog: a manifest with a deliberate stall-then-sleep job → assert
   WEDGED-RETRIED then FAILED-on-second-wedge paths both fire.

## Gates
Watchdog smoke green; existing run_queue behavior unchanged without the flag (summary
byte-stable on the existing fixture manifest); selftest 7/7; medium panel.

## Out of scope
Rewriting the checkpoint format, async-checkpointing migration, touching training math.
Evidence pointers: `logs/*_wedged_ep{1581,162}.{csv,log}` + `jax_ckpt_fast/*_wedged_*` on
gpu-server; wedge tails quoted in `lore/notes_2026-07-09.md` and QUEUE.md.
