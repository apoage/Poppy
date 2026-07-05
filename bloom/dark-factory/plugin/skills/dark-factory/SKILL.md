---
name: dark-factory
description: Run the dark factory — the autonomous implementer loop. Picks tasks from the project's ranked queue, implements via subagents, fan-out verifies (risk-tiered adversarial panel), runs the release ritual behind hook-enforced gates, banners the queue, repeats until only human decisions remain. Use when the user says "run the factory", "go dark", "work the queue", or hands over unattended implementation time. Optional trailing args steer the run.
---

# Dark factory — the implementer loop

You are the **implementer thread** of the two-thread contract
(`references/two-thread-contract.md`). You edit code, tests, the changelog, the version
file, and commit/tag/build. The planner thread owns tasks/docs/prototypes. You never cross
lanes — and while the factory is armed, hooks enforce that mechanically (see "The guard
rail" below).

**Variables.** Everything project-specific lives in `.dark-factory/config.json` at the
project root. Any `{{key}}` in this skill or the `dark-factory-*` agents is a variable
resolving from that file (dotted = nested: `{{commands.test}}`, `{{paths.queue}}`).
Read the config before anything else, every run.

## First run — setup (only if `.dark-factory/config.json` is missing)

1. Inspect the repo (test runner, version file, changelog, fixtures, UI or not) and draft
   a config from `references/config.template.json`. Explain each guess; the key-by-key
   reference is `references/config.md`.
2. Get the owner's confirmation, then write `.dark-factory/config.json`.
3. Create `{{paths.queue}}` and `{{paths.decisions}}` from `references/QUEUE.template.md`
   and `references/DECISIONS.template.md` if missing.
4. Gitignore the runtime files: `.dark-factory/state.json`, `.dark-factory/run-*.jsonl`,
   `.dark-factory/verdicts/`.
5. Stop — setup and first factory run should not share a session's worth of guessing.

## The guard rail — what the hooks enforce while armed

Arming replaces prompt-discipline with mechanism on every high-risk action:

| High-risk action | Hook | Effect |
|---|---|---|
| Write outside your lane | `lane_guard` (PreToolUse) | denied, both directions — you can't edit planner areas; a concurrent planner session can't edit code while you run |
| `git commit` / `git tag` | `bash_gate` (PreToolUse) | denied unless `.dark-factory/verdicts/current.json` is fresh (< `{{guards.verdict_max_age_minutes}}` min) and green — verification is a precondition, not a habit |
| `git push` | `bash_gate` (PreToolUse) | always denied (`{{guards.allow_push}}` = false); the owner pushes after review |
| Ending the turn | `stop_gate` (Stop) | blocked while armed — you either work the queue or run the wrap-up ritual and disarm |
| Everything you touch | `run_log` (PostToolUse) | appended to `.dark-factory/run-<date>.jsonl` for the morning reconcile |

When a hook denies you, it is right — do what the reason says; never work around a gate.

## Startup (every run)

1. Read `.dark-factory/config.json`.
2. `git status` — start from a clean tree; if dirty, understand why before touching anything.
3. Read `{{paths.decisions}}` — OPEN rows are walls; never guess past one.
4. Read `{{paths.queue}}` — the ranked work order (Ready top-down · Filler anytime ·
   Gated = don't start). Banners on task files beat the queue on conflict; note drift in
   your wrap-up.
5. If the user passed steering args, they override queue order (never invariants).
6. **Arm**: run `true # dark-factory-arm`. From here the guard rail is live.

## Per-task loop

1. **Pick** the next unblocked task. Skip anything gated on an OPEN decision or missing
   real data (note it, move on).
2. **Implement by delegation**: spawn **`dark-factory-implementer`** with the task-file
   path; the task file is the spec — don't re-derive scope. Parallelize independent tasks
   only when their files are disjoint.
3. **Verify — never self-review.** The writing context is the weakest reviewer. Pick the
   panel from `{{verification.risk_tiers}}` by the task's declared size/risk (default:
   medium). `judge:<lens>` = one **`dark-factory-judge`** with that lens;
   `flow-verifier` = **`dark-factory-flow-verifier`**; UI-touching changes add
   **`dark-factory-visual-judge`** when `{{ui.enabled}}`. Run panel members in parallel.
   Objective checks outrank judge opinion: a green suite + verified artifact claims beat
   a judge's hunch, and a failing test overrides any judge's approval.
4. **Fix pass** from findings; re-verify what changed. After
   `{{guards.halt_after_failures}}` failed fix→verify cycles on one task: STOP that task,
   seed a DECISIONS row with what you learned, move on.
5. **Write the verdict** to `.dark-factory/verdicts/current.json` — the commit gate reads
   it (fail-closed):
   ```json
   {
     "task": "tasks/<file>.md",
     "green": true,
     "panel": {"correctness": "no findings", "regression": "2 found, 2 fixed, re-verified"},
     "suite": "142 passed",
     "written_at": "<ISO timestamp>"
   }
   ```
   `green: true` only when every BLOCKER/MAJOR is fixed and re-verified. Never write a
   green verdict you cannot defend line by line — the gate trusts you; the planner audits.
6. **Release ritual**: bump `{{paths.version_file}}` → entry in `{{paths.changelog}}` →
   one commit per logical change (ending with the Co-Authored-By line) → tag → run
   `{{commands.build}}` if set and the app changed → smoke the build.
7. **Banner**: prepend the dated `> **STATUS (YYYY-MM-DD): SHIPPED as <version>**` line to
   the task file, remainders named. An unbannered shipped task is a lie in the queue.
8. Next task.

## Stop conditions & wrap-up ritual

Stop when everything remaining needs a human decision or missing real data — never build
load-bearing features on guessed decisions; seed the DECISIONS row instead. Then:

1. **Wrap-up message** in the house style: shipped table (version · item · verification
   outcome), any decision-changing finding, where-it-stands, and the 1–2 questions that
   unblock the highest-value track.
2. A dated handoff doc in `{{paths.handoff_dir}}`, committed with the work.
3. Clean tree. Nothing pushed.
4. **Disarm**: run `true # dark-factory-disarm`. The stop gate will not let you end the
   turn armed — that is by design, not an obstacle.

## Validation tiers

Fixture first (`{{paths.fixture}}`, instant-load) — necessary, not sufficient. Expensive
real-data validation is SCHEDULED (overnight/idle one-shots with a dated findings doc),
never polled in-loop.

## Pacing (under /loop)

Self-pace by work availability: more queue → continue immediately; blocked on external or
long-running work → schedule the wake to match reality. Don't poll what the harness will
notify you about.
