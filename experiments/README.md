# experiments/ — operator → server agent specs

This directory holds **distilled experiment specs** for the server agent to execute.
Workstation-mastered, ships via `tools/deploy/deploy.sh`. Counterpart to
`runs/<id>/` on the server (server-mastered output, pulled back via
`tools/deploy/pull-results.sh`).

## Why distilled, not raw `tasks/TASK_*.md`?

`tasks/` lives on the workstation only — full of cross-references to `lore/`,
dashboard URLs, in-progress thinking, and conversational context. Server agent
doesn't need any of that. A SPEC is the operational subset: **what to run, with
what config, against what success criteria.** No prose.

## Directory layout

```
experiments/<id>/
  SPEC.md         ← required. The contract. Operator-authored.
  config.yaml     ← optional. Run config consumed by the training script.
  inputs/         ← optional. Small static inputs (configs, masks, schedules).
```

Naming: `<id>` is short and stable, e.g. `000_smoke`, `739_dense_sign`,
`741_per_asset_injection`. Match the task number where applicable.

## SPEC.md contract

A SPEC must include, in this order:

1. **One-line goal** — what we're testing.
2. **Source task** — link to `tasks/TASK_*.md` on the workstation if applicable
   (server agent can't follow it; this is for operator audit).
3. **Command** — exact CLI to run, including paths. The server agent runs this
   verbatim; do not include workstation-only flags.
4. **Resources** — GPU count + memory budget, expected wallclock.
5. **Success criteria** — concrete, machine-checkable where possible.
6. **Output expectations** — what files appear in `runs/<id>/` when done.
7. **Notes** — non-obvious gotchas. Keep short.

## How server agent uses these

1. Reads `SPEC.md` start to finish before doing anything.
2. If anything is ambiguous, writes its interpretation to `runs/<id>/notes.md`
   BEFORE starting and stops to wait for clarification on next deploy.
3. Launches the command, capturing stdout/stderr to `runs/<id>/train.log`.
4. On completion (success or fail), writes `runs/<id>/result.md` with success
   criteria status, and a one-line summary of artifacts produced.
5. Server agent never modifies `experiments/<id>/SPEC.md` — that's
   workstation-mastered (per `SERVER.md` authorship invariants).

## Operator workflow

```bash
# Workstation:
mkdir experiments/742_my_idea
$EDITOR experiments/742_my_idea/SPEC.md
tools/deploy/deploy.sh

# Server agent picks it up. Runs. Writes to runs/742_my_idea/.

# Workstation, when ready:
tools/deploy/pull-results.sh runs/742_my_idea
```

## Active specs

See subdirectories. `000_smoke/` is the bring-up sanity check — run that first
on any fresh server env.
