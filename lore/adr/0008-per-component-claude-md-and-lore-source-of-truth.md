---
status: accepted
date: 2026-04-22
deciders: Lukas
---

# 0008. Per-component CLAUDE.md modular structure; `lore/` and `tasks/` are the source of truth

## Status

Accepted.

## Context

Earlier in the project a single top-level CLAUDE.md (and `lore/` files like `DEVELOPMENT_DIARY.md`, `DIARY.md`, `ROADMAP_2026_Q1.md`, `MULTI_ACTION_DESIGN.md`, etc.) tried to be the canonical reference. This produced two failure modes:

1. **Stale top-level docs.** Component facts that changed daily (e.g. encoder version, PopArt range, factored-temperature default) drifted out of sync with code within days. Agents reading the root CLAUDE.md would confidently make decisions on wrong information.
2. **Cross-cutting docs swallowing component specifics.** Architecture decisions for Bella ended up in the Poppy diary; orchestration rules ended up in trader-level files. Searching for "what is the current encoder default" required reading half a dozen files.

The `feedback_read_current_state.md` finding (in user memory) crystallised the rule: top-level CLAUDE.md and any persistent memory must NOT be trusted as the authoritative state. The session-narrative `lore/notes_YYYY-MM-DD.md` files and the per-task `tasks/TASK_XXX_*.md` specs are the source of truth.

## Decision

The canonical structure:

- **Top-level `CLAUDE.md`**: cross-cutting rules only — naming conventions, JAX pitfalls, memory hierarchy philosophy, component pointer table. No component-specific facts.
- **Per-component `CLAUDE.md`** in each subtree (auto-loaded when working in that subtree): all component-specific facts, gotchas, and entry points. One file per component:
  - `mcts_trader/CLAUDE.md` (Poppy)
  - `diff_teacher/CLAUDE.md` (Bella)
  - `tools/orchestration/CLAUDE.md` (Foxglove dojos)
  - `yarrow/CLAUDE.md` (Yarrow)
- **`lore/notes_YYYY-MM-DD.md`**: session narrative. Numbered sections (`§NN`). Captures decisions in motion. **The source of truth for "why".**
- **`tasks/TASK_XXX_*.md`**: feature specs and progress.

**Hard rule** (from `feedback_read_current_state.md` and `CLAUDE.md`): "always check `lore/` and `tasks/` at session start; never rely on stale CLAUDE.md or memory as source of truth for code state."

## Consequences

Gain:
- Component facts are co-located with the code they describe; less drift.
- Top-level CLAUDE.md stays small and stable (cross-cutting rules change rarely).
- Lore section numbering (`§NN`) gives stable cross-document references that survive renames.
- Architecture Decision Records (this directory, `lore/adr/`) extract the load-bearing decisions out of the lore narrative into a searchable index — without losing the narrative itself.

Give up:
- More files to navigate than a single root doc.
- Some duplication is unavoidable (the component pointer table in root CLAUDE.md vs the subtree files).
- Future agents must learn the convention; routine instinct to "read the README" needs replacing with "read the lore + tasks first".

## Alternatives considered

- **One monolithic CLAUDE.md.** The starting state. Rejected after repeated drift incidents.
- **All facts in `lore/` only, no per-component CLAUDE.md.** Rejected because per-component CLAUDE.md is *auto-loaded* by Claude Code when working in that subtree — the auto-load is the load-bearing part.
- **Keep memory as the index.** Rejected — `feedback_read_current_state.md` explicitly downgrades memory.

## References

- Top-level `CLAUDE.md` — the cross-cutting rules + component pointer table.
- `~/.claude/projects/-home-lukas-Trader/memory/feedback_read_current_state.md` (user memory, not in repo) — the load-bearing feedback that triggered this rule.
- `lore/ORGANIZATION_RULES.md` — earlier formulation of the same convention.
- This ADR directory (`lore/adr/`) — extension of the same principle: extract load-bearing decisions out of the time-ordered lore into named, searchable files.
