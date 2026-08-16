# TASK_891 — close the run_queue bare-`python` trap

> **STATUS (2026-08-16): HALTED on the 3-cycle rule (factory run 34) — NOTHING SHIPPED, code preserved
> in `git stash` ("TASK_891 … HALTED on the 3-cycle rule (D42)"). Decision seeded: D42, OPEN.**
> **Do NOT retake this row as written** — it needs D42's re-cut first.
>
> **The trap is still live on gpu-server.** Every failing spelling behaves exactly as it did pre-891, so
> nothing is worse than HEAD; but the two-launch defect (861 chain, 879 rung-3) is NOT yet fixed there.
>
> **Three cycles, three panels, one shape.** (c1) whole-`trader_env/bin` PATH prepend + leading-token
> refusal → `env python`/`nohup python`/`time python`/`bash -c "python …"`/`true && python`/
> `PYTHONPATH=x python` all bypassed the refusal AND got the rejected interpreter, while the blanket
> prepend silently swapped `pip` 23.0.1 → 26.0.1 on rows that never died at HEAD and labelled them
> `passthrough`. (c2) three-class model + word-level regex → `\python` and `python$EMPTY` escaped the
> character class entirely, and `echo python && pip --version` re-triggered the `pip` collision through
> the new `embedded` path. (c3) **orchestrator-directed redesign — a python-only shim dir** (exactly
> `python` + `python3` as `exec` wrappers, NOT symlinks: measured, a symlink shim makes CPython take
> `sys.prefix` from the symlink's own directory, miss `pyvenv.cfg`, and silently load the base pyenv's
> site-packages **while passing every version check**), `$VENV_BIN` never injected, text classification
> demoted to deciding refusal only. That closed both prior MAJORs **structurally**: HEAD-vs-candidate A/B
> job-log diff is exactly ONE hunk, `pip` stays 23.0.1, 11 independently-rebuilt mutations with 0
> survivors, suites 109/109 + 16/16 + 18/18, selftest 12/12.
>
> **WHAT HALTED IT.** A manifest row that assigns `PATH=` itself (`PATH=/dir python …`, `env -i PATH=…
> python …`) locally shadows the injected shim — bash performs the per-command assignment **before** the
> path search — so the job runs the ambient interpreter, exits `rc=0`, and prints a **FALSE**
> `interpreter=venv-path:…` label claiming the venv resolved it, which also hides it from a pre-launch
> `--dry-run` review. Orchestrator-verified first-hand, not on report.
>
> **The pattern IS the finding:** the escaping spelling got more exotic each cycle
> (`env python` → `\python` → `PATH= python`). This row's contract — *no manifest spelling may silently
> run the wrong interpreter* — is a claim about arbitrary shell text, and wrapping cannot close it.
> D42(a) recommends re-cutting to an honest, bounded contract and shipping c3's shim behind it, with the
> label never asserting a resolution the code cannot back.

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
