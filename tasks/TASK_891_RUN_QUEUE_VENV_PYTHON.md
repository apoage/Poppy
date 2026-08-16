# TASK_891 — close the run_queue bare-`python` trap

**Status**: READY 2026-08-16 (planner/orchestrator; recommended unspecced since 2026-08-06 — it has
now cost TWO launches).
**Size: XS. Risk: LOW** — scheduler plumbing in `tools/deploy/`, no numerics, no canon surface.
**Parallel-safe FILLER**: touches `tools/deploy/run_queue.sh` (+ its selftest/smoke) only — zero
overlap with every other Ready row (886/890/878/881 are all `experiments/`+`mcts_trader/` surface).

## The defect

`tools/deploy/run_queue.sh` launches each manifest line via `bash -c` from the repo root and does
**nothing** about the Python interpreter — no venv activation, no PATH prepending, no check. On
gpu-server a bare `python` resolves to **python2**, so a manifest whose commands start with
`python …` has every job die in ~30 s with a PEP-263 SyntaxError. This has now burned two real
launches (the 861 chain, then the 879 rung-3 launch; the p1 ops note in QUEUE.md says the same of
fresh tmux sessions). Every historical manifest in the tree uses bare `python` — the trap is the
default spelling, not an exotic one.

## Scope

1. **Resolve the repo venv in the runner.** Before launching any job, `run_queue.sh` resolves
   `<repo>/trader_env/bin/python` and exports/prepends the venv's `bin/` on the job's PATH (or
   rewrites a leading bare `python`/`python3` token to the resolved absolute path), so a manifest
   written with bare `python` runs the venv interpreter regardless of the tmux environment the
   queue was started in.
2. **Fail loud, never silently substitute.** If the venv python is absent or not executable, the
   queue refuses the manifest at parse time with a named error (job id + line) — it does NOT fall
   back to whatever `python` is on PATH. A pre-launch version check (the resolved interpreter
   reports `sys.version_info.major == 3`) fires the same refusal.
3. **`--dry-run` shows the resolution.** The schedule printout names the interpreter each job
   will run, so the refusal/resolution is visible before anything launches.
4. **Header comment updated** — the manifest-format section must state the interpreter contract
   (bare `python` = repo venv; anything else must be an absolute path or an explicit
   `source …/activate && …` in the line itself).

## Acceptance (gates)

- (i) On a PATH where `python` is python2 (simulate with a stub dir prepended to PATH — do NOT
  touch the host's real python), a manifest line `j1 cpu 4G python -c "import sys;print(sys.version)"`
  runs python3 from `trader_env` and the job log shows it.
- (ii) With `trader_env` renamed away (sandboxed copy of the repo — not the real tree), the same
  manifest is REFUSED at parse time, naming the job and the missing interpreter, exit non-zero,
  no job launched.
- (iii) `--dry-run` output names the resolved interpreter per job.
- (iv) A manifest using an absolute interpreter path is passed through untouched.
- (v) Default behaviour otherwise byte-identical: the summary table and log layout are unchanged
  when the venv resolves cleanly (the trap fires only where jobs previously DIED).
- (vi) Regression tests for (i)–(v), mutations independently rebuilt.

## Out of scope

`safe_run.sh` semantics, the stall watchdog, GPU scheduling. Rewriting historical manifests under
`runs/` (they are records — the fix makes their spelling work, which is the point). Server-side
deployment (the planner redeploys `tools/deploy/` to gpu-server after ship, as with TASK_833).
