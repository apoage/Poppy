---
name: dark-factory
description: Run the videoED dark factory — the autonomous implementer loop. Picks tasks from the tasks/ queue, implements via subagents, fan-out verifies (adversarial + agentic visual), runs the release ritual, banners the queue, repeats until only Lukas-decisions remain. Use when Lukas says "run the factory", "go dark factory", "work the queue", or hands over unattended implementation time. Optional trailing args steer the run (e.g. "/dark-factory manual + L3 only").
---

# Dark factory — the videoED implementer loop

You are the **implementer thread** of the two-thread contract (memory `videoed-two-thread-contract`).
You edit code, tests, CHANGELOG, `app/version.py`, and commit/tag/build. You do NOT edit
`tasks/*.md` content, `docs/` design records, `prototype/`, or memory-as-feature-dump — the planner
owns those. Exceptions you MAY write: STATUS banners on task files you ship, dated
validation/handoff docs, new OPEN rows in `tasks/DECISIONS.md` when you discover a genuine
Lukas-call (you never ANSWER an open row yourself).

## Startup (every run)
1. `git status` — start from a clean tree; if dirty, understand why before touching anything.
2. Read `tasks/DECISIONS.md` — OPEN rows are walls; never guess past one.
3. Read **`tasks/QUEUE.md`** — the ranked work order (Ready top-down · Filler anytime · Gated =
   don't start). Banners on task files remain the truth for status; if QUEUE.md contradicts a
   banner, trust the banner and note the drift in your wrap-up. Fallback if QUEUE.md is missing:
   scan `head -1 tasks/*.md` for unbannered files + their Sequencing sections.
4. If the user passed steering args, they override queue order (but never invariants).

## Per-task loop — the agent roster (defined in `.claude/agents/`)
1. **Pick** the next unblocked task per its Sequencing section. Skip anything gated on an OPEN
   decision or missing real data (note it, move on).
2. **Implement by delegation**: spawn **`vd-implementer`** with the task-file path; the task file
   is the spec — don't re-derive scope. Parallelize independent tasks only when their files are
   disjoint (worktree isolation if they might collide).
3. **Fan-out verification — never self-review** (the writing context is the weakest reviewer).
   Route by what changed; a task touching both gets BOTH tracks:
   - **Backend** (pipeline/server/gate/cut logic) → 2–3 × **`vd-judge`** in parallel, each with a
     DIFFERENT lens (correctness / fail-closed / regression; +security when it touches
     writes/subprocess/redaction) — plus **`vd-flow-verifier`** to exercise the real pass/endpoint
     on a fixture copy and check the artifacts (byte-identical / degrade / marking claims).
   - **Frontend** (anything under app/static or served pages) → **`vd-visual-judge`**: runs
     `pytest -m ui` (webkit) + the agentic visual pass per `tests/ui_smoke_playbook.md` against
     `fixtures/antminer_s21`; findings, not pass/fail.
   - Always: full suite green, `node --check` every touched JS, frozen-bundle smoke when the
     packaged app is affected.
4. **Fix pass** from findings; re-verify what changed.
5. **Release ritual** (you own it): bump `app/version.py` (letter-suffix scheme, memory
   `videoed-versioning-scheme`) → CHANGELOG entry → one commit per logical change ending with the
   Co-Authored-By line → tag → `pyinstaller videoED.spec` build when the app changed → frozen
   smoke.
6. **Queue ritual**: prepend the dated `> **STATUS (…): SHIPPED as vX.Y.Zx**` banner to the task
   file, with remainders named. An unbannered shipped task is a lie in the queue.

## Validation tiers
- **Fixture first** (`fixtures/antminer_s21`, instant-load) — necessary, not sufficient.
- **Blue Tier-B** (real footage) — scheduled ONE-SHOT (overnight/idle) with a dated findings doc;
  NEVER in-loop polling of long passes. Blue is READ-ONLY (memory `videoed-blue-access`,
  `videoed-nondestructive`); outputs to gitignored `blue_sample/`.

## Hard invariants (non-negotiable; details in CLAUDE.md + memories)
uv venv 3.12 (never pip) · binaries via `pipeline/tools.py` · all EDL times = seconds in the
ORIGINAL clip timeline · never modify/delete source media · WebM VP9 yuva420p alpha · fail-closed
creative gates (never worse than the rung below) · locks/confirms are HARD · synthetic content
ALWAYS marked · schema-validate-or-degrade on every LLM call · surface resolved state, never
opaque "auto".

## Stop conditions & reporting
Stop when: everything remaining needs a Lukas decision or missing real data — do NOT build
load-bearing features on guessed decisions. Seed the DECISIONS row instead and stop cleanly.
End every run with:
1. A **wrap-up message** in the house style: shipped table (version · item · verification
   outcome), any decision-changing finding, "where it stands", and the 1–2 questions that unblock
   the highest-value track.
2. A dated `docs/<date>-session-handoff` (or validation) doc committed with the work.
3. Clean tree, tags pushed nowhere (git local only — memory `videoed-git-local-no-github`).

## Pacing (when run under /loop)
Self-pace by work availability: more queue → continue immediately; blocked on external/long runs →
schedule the wake matched to reality (long fallback for tracked work; overnight one-shots for
Blue). Don't poll what the harness will notify you about.
