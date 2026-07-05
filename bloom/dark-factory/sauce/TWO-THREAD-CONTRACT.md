# The two-thread contract — the sauce that makes the factory work

The dark factory is NOT a while-true loop with a todo list. It is one half of a two-session
working model, and the separation is load-bearing. This document is the generalized contract;
the skill (`skills/dark-factory/SKILL.md`) is its executable form for the implementer side.

## The two threads

**Thread 1 — the PLANNER** (interactive session, human in the loop)
- Writes ONLY: `tasks/`, `docs/`, `prototype/`, agent/skill definitions, memory.
- Designs features, grooms the queue, drafts mocks, consolidates decisions.
- When it finds a bug it documents the repro + verified fix recipe in a task file —
  it NEVER applies the fix. (This rule exists because of a real edit race: planner patched a
  file the implementer owned, both had to untangle a revert.)

**Thread 2 — the IMPLEMENTER** (autonomous loop, unattended)
- Owns: code, tests, CHANGELOG, version file, commits/tags/builds — the whole release ritual.
- Never edits task-file CONTENT, design docs, prototypes, or memory-as-feature-dump.
- Allowed writes into planner territory: STATUS banners on tasks it ships, dated
  handoff/validation docs, NEW open rows in DECISIONS.md when it hits a genuine human call.

Single-writer everywhere: for any file, exactly one thread may write it. That's the whole trick —
two agents on one repo without merge hell.

## The three shared surfaces

1. **`tasks/QUEUE.md`** — the ranked work order (planner-maintained). Ready top-down, Filler
   anytime, Gated don't-start. The factory takes from the top. Banners on task files beat
   QUEUE.md on conflict (the queue is an index, banners are the truth).
2. **STATUS banners** — dated `> **STATUS (YYYY-MM-DD): SHIPPED as vX.Y.Z …**` prepended to a
   task file's line 1 when it ships or is superseded, always mapped to a CHANGELOG entry, never
   guessed. An unbannered shipped task is a lie in the queue. Planner sweeps and verifies
   banners against code after each factory run.
3. **`tasks/DECISIONS.md`** — the single blocked-on-human surface. One row per call,
   recommendation first. OPEN rows are WALLS: the loop never guesses past one, the planner never
   scatters open questions elsewhere. Answered rows move to DECIDED with the date.

## Verification doctrine — never self-review

The context that wrote a change is the weakest possible reviewer of it. Every load-bearing
change gets fan-out verification by SEPARATE subagents:

- 2–3 × **adversarial judges** in parallel, each with ONE named lens (correctness /
  fail-closed / regression / security). Refutation-first: a judge wins by breaking the change.
  An empty verdict must list the attacks attempted, or it's worthless.
- A **flow verifier** that runs the real pass/endpoint on a fixture COPY and inspects the
  artifacts — claims in the task file are its test cases ("byte-identical when off" → diff it).
- A **visual judge** for anything UI: deterministic smoke layer first, then an agentic pass —
  screenshots JUDGED against the task file + the project's design principles. Findings ranked
  by severity, never a bare pass/fail (screenshot judgment is noisy; evidence travels with
  every finding).

## Validation tiers

- **Fixture first** — a small baked project that loads instantly. Necessary, not sufficient.
- **Real-data runs are SCHEDULED, never polled** — big-footage validation goes overnight/idle
  as a one-shot with a morning findings doc. An autonomous loop polling a 28-minute pass is
  burned context for nothing.

## Stop conditions

The factory stops CLEANLY when everything remaining needs a human decision or missing real
data. It does not build load-bearing features on guessed decisions — it seeds the DECISIONS
row and stops. Every run ends with a wrap-up (shipped table, decision-changing findings,
where-it-stands, the 1–2 questions that unblock the most) and a clean tree.

## Why this beats naive looping

- Single-writer areas → zero edit races between concurrent sessions.
- Queue + banners → the backlog is always truthful; no stall tasks, no re-derived scope.
- DECISIONS walls → the loop can run unattended for hours without inventing product calls.
- Fan-out verification → the loop's throughput doesn't come at correctness's expense.
- Scheduled real-data runs → unattended time is spent building, not waiting.
