# TASK_891 — close the run_queue bare-`python` trap (RE-CUT per D42)

> **STATUS (2026-08-16, planner): READY — RE-CUT per D42(a), operator-ruled.** The run-34 halt was
> correct: the original contract ("no manifest spelling may silently run the wrong interpreter") is
> a claim about arbitrary shell text and wrapping cannot close it. This re-cut ships cycle-3's shim
> behind an HONEST BOUNDED contract. **Resume from tag `stash-891-halted-d42`** — cycle-3's shim
> (python-only shim dir, exec wrappers NOT symlinks) was verified at halt time: 11 independently-
> rebuilt mutations 0 survivors; suites 109/109 + 16/16 + 18/18; selftest 12/12; HEAD-vs-candidate
> A/B job-log diff exactly one hunk; `pip` stays 23.0.1.
>
> _(Prior banner, superseded: HALTED on the 3-cycle rule (factory run 34) — full post-mortem in
> `lore/handoff_2026-08-16_factory_run_34.md` and D42.)_

**Size: XS/S. Risk: LOW** — scheduler plumbing in `tools/deploy/`, no numerics, no canon surface.
**Parallel-safe FILLER**: touches `tools/deploy/run_queue.sh` (+ its tests) only.

## The bounded contract (D42(a) — this replaces the run-34 contract in full)

1. **A leading bare `python`/`python3` token** (first whitespace-separated word of the command)
   resolves to the repo venv via a **python-only shim dir** prepended to the job's PATH: a
   directory containing exactly two `exec` wrappers (`python`, `python3`) that exec
   `trader_env/bin/python`. NOT symlinks — measured in run 34: a symlink shim makes CPython take
   `sys.prefix` from the symlink's own directory, miss `pyvenv.cfg`, and silently load the base
   pyenv's site-packages while passing every version check. The shim dir contains NOTHING else, so
   no other binary (`pip`, …) is shadowed.
2. **Every other spelling passes through UNTOUCHED and is labelled `passthrough`** — `env python`,
   `bash -c "python …"`, `PATH=/x python …`, `sudo python`, anything. The queue makes NO claim
   about them. The pre-launch `--dry-run` printout and the launch log say `interpreter=venv-shim`
   or `interpreter=passthrough:<token>` — and **never a resolution claim the code cannot back**.
   The run-34 halt case (`PATH=… python …` locally shadows the shim) must PRINT `passthrough`,
   not `venv-path:…`.
3. **The documented-limits list is derived from the actual mechanism**, not from aspiration: the
   header comment states precisely the one spelling class that is covered (leading bare token)
   and that everything else is the manifest author's responsibility.
4. **Fail loud on a broken venv**: if `trader_env/bin/python` is missing/not executable/not
   python3, a job whose leading token is bare `python`/`python3` is REFUSED at parse time (named
   job + line, non-zero exit, nothing launched for that row). Passthrough rows are unaffected.

## Acceptance (gates)

- (i) On a PATH where `python` is python2 (stub dir prepended — do NOT touch the host's real
  python), a manifest line `j1 cpu 4G python -c …` runs the venv's python3 and the log says so.
- (ii) `env python …` / `bash -c "python …"` / `PATH=/stub python …` rows run the AMBIENT
  interpreter and are labelled `passthrough` in both `--dry-run` output and the launch log —
  including the `PATH=`-shadow case that halted run 34.
- (iii) Venv missing (sandboxed copy, `trader_env` renamed): bare-python rows refused by name,
  exit non-zero, passthrough rows still scheduled.
- (iv) No binary other than `python`/`python3` is shadowed: `pip --version` in a manifest prints
  the same version with the queue's PATH injection as without.
- (v) Default behaviour otherwise byte-identical (summary table, log layout) when no bare
  `python` rows are present.
- (vi) Every guard has a test that goes red when the guard is removed at its REAL drop site
  (run-34's standing lesson); mutations independently rebuilt; no test stubs the mechanism
  under test.

## Out of scope

`safe_run.sh` semantics, the stall watchdog (its load-sensitive flake is a separate noted
candidate row), GPU scheduling. Rewriting historical manifests. Server-side deployment — the
planner redeploys `tools/deploy/` to gpu-server after ship, as with TASK_833.
