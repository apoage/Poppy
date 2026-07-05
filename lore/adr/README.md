# Architecture Decision Records

Load-bearing architectural decisions, extracted from the session-narrative `lore/notes_YYYY-MM-DD.md` files into standalone, searchable records. These live under `lore/` (not `docs/`) because they were born inside the lore narrative and are most useful when read alongside it.

A decision belongs here only if it meets one of these bars:

- **Contested** — alternatives were on the table and one was rejected.
- **Non-obvious** — the rationale isn't visible from the code.
- **Load-bearing** — reversing it would force a major rewrite or invalidate prior baselines.

Routine implementation choices belong in code, in task specs (`tasks/TASK_XXX_*.md`), or in the lore narrative itself.

## MADR template

When adding a new ADR, copy this skeleton:

```markdown
---
status: accepted    # or: superseded, deprecated
date: YYYY-MM-DD    # the day the decision was made
deciders: Lukas
superseded-by: NNNN # optional, only if status is superseded
---

# NNNN. Short title

## Status

One line.

## Context

What forced the decision. What evidence drove it. Cite lore sections by file + section, e.g. `lore/notes_2026-04-22.md §49`.

## Decision

What was decided, in declarative voice.

## Consequences

What we gain. What we give up. What becomes harder.

## Alternatives considered

(Optional — only if the alternatives are documented in lore.)

## References

Lore sections, task files, papers, related ADRs.
```

If the rationale is genuinely uncertain, replace `## Decision` with `## Decision uncertainty` and say so explicitly. Do not fabricate rationale.

Naming: `NNNN-short-kebab-title.md`, four-digit zero-padded id. Increment monotonically; never re-use an id.

## Index

| ID | Title | Status |
|---|---|---|
| [0001](0001-three-phase-exit-pipeline.md) | Three-phase Expert Iteration pipeline (Foxglove → Poppy) | accepted |
| [0002](0002-trm-muzero-no-attention-reasoning.md) | TRM + MuZero as the planner architecture; no attention in reasoning or dynamics | accepted |
| [0003](0003-encoder-v3-32ch-cross-sectional.md) | Encoder V3 (minGRU temporal pool, 32 channels, embed_dim=64, heads=8) | accepted |
| [0004](0004-oracle-modes-topk-densesign.md) | Oracle modes — topk_path as default, dense_sign retained as diagnostic tool | accepted |
| [0005](0005-dir-acc-ceiling-decomposition.md) | The 0.57 dir_acc ceiling is decomposed ~50 % architectural, ~50 % informational | accepted |
| [0006](0006-stage-b-value-weight-required.md) | Stage B requires `--value-weight ≥ 0.5` — value training is mandatory for ExIt | accepted |
| [0007](0007-film-injection-post-temporal-pool.md) | Per-asset scalar injection via FiLM, post-temporal-pool (TASK_741, codename Mint) | accepted |
| [0008](0008-per-component-claude-md-and-lore-source-of-truth.md) | Per-component CLAUDE.md modular structure; `lore/` and `tasks/` are the source of truth | accepted |
| [0009](0009-apothecary-naming-theme.md) | Apothecary / medicinal-plants naming theme for components | accepted |
