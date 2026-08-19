# TASK_891 — close the run_queue bare-`python` trap (SECOND RE-CUT per D46)

> **STATUS (2026-08-19): SHIPPED as `task891-run-queue-venv-python` (`706ce97`), factory run 36,
> 3 cycles (cycles 2-3 on opus).** D46's absolute-path substitution, unchanged. Gates (i)-(vii) met.
> Suites: interpreter **25/25** (12 gates + 13 mutation kills), scheduler 16/16, watchdog 18/18,
> selftest 12/12, gate (v) byte-identity vs HEAD.
> **The defect was the CLAIM, and the claim is what changed.** A fail-closed judge demonstrated live
> that the parse-time-only venv check left an hours-long TOCTOU while the code asserted there was
> none; fixed by a launch-time re-probe (one shared `_venv_usable_ok` at both sites) plus text that
> says what is true — NOT a security boundary (same-uid `systemd-run --user --scope`, an ordinary
> same-user-writable interpreter, and a `protected` list with no enforcement code), so the checks buy
> protection against ACCIDENT, not against a hostile process. Probe limits MEASURED: four forgeries
> refuse, `exec -a <our path> <real python>` + a planted `pyvenv.cfg` passes, and the comment says so.
> **Found in the verification machinery itself:** the suite printed 18 passed while M8's mutant was
> never built (anchor drift scored as a KILL). A build failure now FAILS the suite by name — proven on
> both mutation paths, all 12 anchors re-verified unique.
> **Withdrawn against its own evidence:** "freed exactly once" — a double-`_free_gpu` mutant SURVIVED
> (duplicate pool entry skipped by the per-GPU `flock`), so the gate claims only "came back, never
> double-booked".
> **Remainders → planner:** redeploy `tools/deploy/` to gpu-server (as TASK_833);
> `runs/860_codistill/manifest_860.txt` stays UNTRACKED so its inline comment is non-durable — the
> durable half is the runtime chained-python warning, and the tracking question is **D48**; the
> watchdog suite's load-sensitive flake remains a separate candidate row (did not fire this run).
>
> _(Prior banners, superseded: run-34 HALT (D42) and run-35 HALT (D46) — post-mortems in
> `lore/handoff_2026-08-16_factory_run_34.md`, the run-35 banner history, and DECISIONS D42/D46.)_

> **STATUS (2026-08-17, planner): READY — RE-CUT per D46, operator-RATIFIED 2026-08-17.** D42(a)'s
> shim mechanism is ABANDONED: run 35 proved the shim dir exists only to serve the PATH prepend, and
> the prepend is itself the attack surface (`env … bash -c` execvp's `bash`, and the wrapper's own
> `#!/usr/bin/env bash` shebang, both resolved through the already-prepended job-writable PATH).
> **The mechanism is now ABSOLUTE-PATH SUBSTITUTION and nothing else.** Resume from tag
> `stash-891r-halted-d46`: both original kills verified DEAD there (fake-venv refused at parse time;
> mid-wait shim breakage refused three ways incl. a forged probe), 6/6 rebuilt mutations red, suites
> 16/16 + 16/16 + 18/18 + selftest 12/12. **The three-tier honest contract (CLAIMED /
> BEST-EFFORT-NOT-CLAIMED / NOT COVERED) survives verbatim** — it was the best-written artifact of
> run 35.
>
> _(Prior banners, superseded: run-34 HALT (D42) and run-35 HALT (D46) — full post-mortems in
> `lore/handoff_2026-08-16_factory_run_34.md`, the run-35 banner history, and DECISIONS D42/D46.)_

**Size: XS/S. Risk: LOW** — scheduler plumbing in `tools/deploy/`, no numerics, no canon surface.
**Parallel-safe FILLER**: touches `tools/deploy/run_queue.sh` (+ its tests) only.

## The bounded contract (D46, operator-ratified — replaces the D42(a) shim contract)

1. **Claim exactly one spelling class: a leading bare `python`/`python3` token** (first
   whitespace-separated word of the command). Deliver it by **absolute-path substitution**: the
   command is rewritten to `<repo>/trader_env/bin/python <rest>` before launch. **No shim
   directory, no PATH prepend, no wrapper scripts** — nothing job-writable enters the trust path.
2. **Every other spelling passes through byte-untouched and is labelled `passthrough`** —
   including chained/child pythons (`bash -c "python …"`, `env python`, `python a.py && python
   b.py`'s second command), which are a documented **NOT COVERED** limit. The three-tier contract
   text (CLAIMED / BEST-EFFORT-NOT-CLAIMED / NOT COVERED) is carried over from the halted work
   verbatim, with the limits re-derived from THIS mechanism.
3. **Fail loud on a broken venv**: if `trader_env/bin/python` is missing/not executable/not
   python3, a claimed-spelling row is REFUSED at parse time (named job + line, non-zero exit).
   Passthrough rows are unaffected.
4. **`--dry-run` and the launch log print the rewritten command** for claimed rows — the operator
   sees the absolute path, not a description of one.
5. **One-time manifest sweep, in scope**: tracked manifests containing chained/child `python`
   spellings (`runs/860_codistill/manifest_860.txt:6-8` is the named shape) get an inline comment
   or respelling so no tracked manifest silently relies on NOT-COVERED behaviour. This is a
   comment/respelling pass only — no semantics change to any manifest that was already correct.

## Acceptance (gates)

- (i) On a PATH where `python` is python2 (stub dir prepended — do NOT touch the host's real
  python), a manifest line `j1 cpu 4G python -c …` runs the venv's python3 and the log says so.
- (ii) `env python …` / `bash -c "python …"` / `PATH=/stub python …` rows run the AMBIENT
  interpreter and are labelled `passthrough` in both `--dry-run` output and the launch log —
  including the `PATH=`-shadow case that halted run 34.
- (iii) Venv missing (sandboxed copy, `trader_env` renamed): bare-python rows refused by name,
  exit non-zero, passthrough rows still scheduled.
- (iv) **No PATH manipulation at all**: a claimed row's environment is byte-identical to a
  passthrough row's (`env` diff empty); `pip --version` inside any job resolves exactly as the
  ambient shell does. The halted shim's `cmp`/probe machinery is GONE — deleted, not bypassed.
- (v) Default behaviour otherwise byte-identical (summary table, log layout) when no bare
  `python` rows are present; the `--dry-run` printout shows the rewritten absolute-path command
  for claimed rows and `passthrough:<token>` for the rest.
- (vi) The manifest sweep lands: every tracked manifest either contains no NOT-COVERED spelling
  or carries an inline comment naming it; the sweep diff is comment/respell only.
- (vii) Every guard has a test that goes red when the guard is removed at its REAL drop site
  (run-34/35 standing lesson); mutations independently rebuilt; no test stubs the mechanism
  under test.

## Out of scope

`safe_run.sh` semantics, the stall watchdog (its load-sensitive flake is a separate noted
candidate row), GPU scheduling. Rewriting historical manifests. Server-side deployment — the
planner redeploys `tools/deploy/` to gpu-server after ship, as with TASK_833.
