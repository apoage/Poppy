# The two-thread contract — the sauce that makes the factory work

The dark factory is NOT a while-true loop with a todo list. It is one half of a
two-session working model, and the separation is load-bearing. This document is the
doctrine; `SKILL.md` is its executable form for the implementer side; the plugin's hooks
are its enforcement.

## The two threads

**Thread 1 — the PLANNER** (interactive session, human in the loop)
- Writes ONLY: the task queue, design docs, prototypes, agent/skill definitions, config.
- Designs features, grooms the queue, drafts mocks, consolidates decisions.
- When it finds a bug it documents the repro + verified fix recipe in a task file —
  it NEVER applies the fix. (This rule exists because of a real edit race: a planner
  patched a file the implementer owned, and both had to untangle a revert.)

**Thread 2 — the IMPLEMENTER** (the factory: autonomous loop, unattended)
- Owns: code, tests, changelog, version file, commits/tags/builds — the whole release
  ritual.
- Never edits task-file CONTENT, design docs, or prototypes.
- Allowed writes into planner territory: STATUS banners on tasks it ships, dated
  handoff/validation docs, NEW open rows in DECISIONS when it hits a genuine human call
  (it never ANSWERS an open row itself).

Single-writer everywhere: for any file, exactly one thread may write it. That's the whole
trick — two agents on one repo without merge hell.

## Enforcement: hooks, not honor

The first version of this contract was prompt-discipline, and prompt-discipline drifts —
a tired session "just fixes one typo" out of lane and nothing catches it. The plugin
turns each high-risk rule into a runtime guard the loop cannot ignore:

| Rule | Mechanism |
|---|---|
| Single-writer lanes | `lane_guard` PreToolUse hook — denies out-of-lane writes in BOTH directions while armed, keyed to the arming session's id |
| No commit without verification | `bash_gate` — `git commit`/`git tag` require a fresh green `.dark-factory/verdicts/current.json`; fail closed |
| Never push | `bash_gate` — `git push` denied outright while armed |
| Clean exit ritual | `stop_gate` — the armed session cannot end its turn without bannering, handing off, cleaning the tree, and disarming |
| Morning reconcile | `run_log` — every mutation appended to `.dark-factory/run-<date>.jsonl` |

Convention is what the model reads; hooks are what the model hits. Ship both.

## The three shared surfaces

1. **The queue** (`{{paths.queue}}`) — the ranked work order (planner-maintained). Ready
   top-down, Filler anytime, Gated don't-start. The factory takes from the top. Banners
   on task files beat the queue on conflict (the queue is an index, banners are the truth).
2. **STATUS banners** — dated `> **STATUS (YYYY-MM-DD): SHIPPED as <version> …**`
   prepended to a task file's line 1 when it ships or is superseded, always mapped to a
   changelog entry, never guessed. An unbannered shipped task is a lie in the queue.
   Planner sweeps and verifies banners against code after each factory run.
3. **DECISIONS** (`{{paths.decisions}}`) — the single blocked-on-human surface. One row
   per call, recommendation first. OPEN rows are WALLS: the loop never guesses past one,
   the planner never scatters open questions elsewhere. Answered rows move to DECIDED
   with the date.

## Verification doctrine — never self-review, objective checks first

The context that wrote a change is the weakest possible reviewer of it. Every
load-bearing change gets fan-out verification by SEPARATE subagents — sized to risk, not
a constant:

- **Objective checks outrank judges.** The test suite, artifact diffs, schema checks, and
  the flow verifier's real-execution evidence are ground truth; LLM judgment covers only
  the ground objective checks can't reach (intent, design conformance, visuals).
- **Adversarial judges** in parallel, each with ONE named lens (correctness / fail-closed
  / regression / security). Refutation-first: a judge wins by breaking the change. An
  empty verdict must list the attacks attempted, or it's worthless.
- **A flow verifier** that runs the real pass/endpoint on a fixture COPY and inspects the
  artifacts — claims in the task file are its test cases ("byte-identical when off" →
  diff it).
- **A visual judge** for anything UI: deterministic smoke layer first, then an agentic
  pass — screenshots JUDGED against the task file + the project's own design principles.
  Findings ranked by severity with evidence attached, never a bare pass/fail.

## Validation tiers

- **Fixture first** — a small baked project that loads instantly. Necessary, not
  sufficient.
- **Real-data runs are SCHEDULED, never polled** — expensive validation goes
  overnight/idle as a one-shot with a morning findings doc. An autonomous loop polling a
  28-minute pass is burned context for nothing.

## Stop conditions

The factory stops CLEANLY when everything remaining needs a human decision or missing
real data. It does not build load-bearing features on guessed decisions — it seeds the
DECISIONS row and stops. Every run ends with a wrap-up (shipped table, decision-changing
findings, where-it-stands, the 1–2 questions that unblock the most), a clean tree, and a
disarm. A task that fails its fix→verify cycle `{{guards.halt_after_failures}}` times
HALTs to a DECISIONS row instead of burning the night on it.

## Why this beats naive looping

- Single-writer areas (hook-enforced) → zero edit races between concurrent sessions.
- Queue + banners → the backlog is always truthful; no stale tasks, no re-derived scope.
- DECISIONS walls → the loop runs unattended for hours without inventing product calls.
- Fan-out verification behind a fail-closed commit gate → throughput never comes at
  correctness's expense.
- Scheduled real-data runs → unattended time is spent building, not waiting.

## Minimum viable factory

SKILL.md + `dark-factory-implementer` + ONE `dark-factory-judge` (correctness lens) + a
queue file gets you 80% of the value. Add the flow verifier when your app writes
artifacts worth inspecting, the visual judge when you have a UI, and tighten the risk
tiers as the stakes grow.
